import fs from "fs/promises";
import EventEmitter from "events";
import path from "path";
import { performance } from "perf_hooks";
import axios from "axios";
import chalk from "chalk";
import dayjs from "dayjs";
import sharp from "sharp";
import { main } from "./main";
import {
  API_URL,
  DEFAULT_BATCH_EXCLUDED_PATHS,
  DEFAULT_BATCH_SIZE,
  DEFAULT_HIRES_DENOISING_STRENGTH,
  DEFAULT_HIRES_SCALE,
  DEFAULT_HIRES_UPSCALER,
  DEFAULT_LORA_MODEL,
  DEFAULT_LORA_PARAMS_PATH,
  DIR_NAMES,
  LORA_TRAINING_PARAMS_FILE_NAME,
  OUTPUT_DIR,
  REPROD_DIFF_TOLERANCE,
  TXT2IMG_OVERRIDES_FILE_NAME,
  VAE_DIR,
} from "./constants";
import {
  FileNames,
  ImageFileNames,
  LoraTrainingParams,
  Model,
  NoEmit,
  ParamFileNames,
  Txt2ImgOverrides,
} from "./types";
import {
  Image,
  PromiseQueue,
  chunkArray,
  compareImage,
  convertImagesToJPG,
  delimit,
  dirToFilePaths,
  doesPathExist,
  escapeRegEx,
  getImagesInFolders,
  makeConsoleList,
  prompt,
  randomSort,
  round,
  sha256File,
} from "./utils";

/* -------------------------------------------------------------------------- */
/*                                    UTILS                                   */
/* -------------------------------------------------------------------------- */
/* ------------------------------ Main Emitter ------------------------------ */
export const mainEmitter = new EventEmitter();

mainEmitter.on("done", () => {
  console.log(chalk.grey("-".repeat(100)));
  main();
});

/* ----------- Sequential Generation for Upscaling / Reproduction ----------- */
export class GenerationQueue extends PromiseQueue {
  activeModel: string = null;
  activeVAEHash: string = null;

  getActiveModel() {
    return this.activeModel;
  }

  getActiveVAEHash() {
    return this.activeVAEHash;
  }

  getOutputDir() {
    return path.join(OUTPUT_DIR, dayjs().format("YYYY-MM-DD"));
  }

  setActiveModel(model: string) {
    this.activeModel = model;
  }

  setActiveVAEHash(vae: string) {
    this.activeVAEHash = vae;
  }
}

/* ---------------------------- Get Active Model ---------------------------- */
export const getActiveModel = async (withConsole = false) => {
  const config = (await axios({ method: "GET", url: `${API_URL}/options` }))?.data;
  const model = (config?.sd_model_checkpoint as string)
    .replace(/\\/gi, "_")
    .replace(/(\.(ckpt|checkpoint|safetensors))|(\[.+\]$)/gi, "")
    .trim();
  if (withConsole) console.log("Active model: ", chalk.cyan(model));
  return model;
};

/* ---------------------------- Get Active VAE ---------------------------- */
export const getActiveVAE = async (withConsole = false) => {
  const config = (await axios({ method: "GET", url: `${API_URL}/options` }))?.data;
  const vae = (config?.sd_vae as string).trim();
  if (withConsole) console.log("Active VAE: ", chalk.cyan(vae));
  return vae;
};

/* -------------- Get Image and Generatiom Parameter File Names ------------- */
export const listImageAndParamFileNames = async (withRecursiveFiles = false) => {
  console.log(`Reading files${withRecursiveFiles ? " (recursively)" : ""}...`);

  const regExFilter = new RegExp(
    Object.values(DIR_NAMES)
      .filter((dirName) => ![DIR_NAMES.nonUpscaled, DIR_NAMES.reproducible].includes(dirName))
      .map((dirName) => escapeRegEx(dirName))
      .join("|"),
    "im"
  );

  const files = (await (withRecursiveFiles ? dirToFilePaths(".") : fs.readdir("."))).filter(
    (filePath) => !regExFilter.test(filePath)
  );

  const [imageFileNames, paramFileNames] = files.reduce(
    (acc, cur) => {
      const fileName = cur.substring(0, cur.lastIndexOf("."));
      const ext = cur.substring(cur.lastIndexOf("."));
      if (ext === ".jpg") acc[0].push(fileName);
      else if (ext === ".txt") acc[1].push(fileName);
      return acc;
    },
    [[], []] as string[][]
  );

  console.log(
    `${chalk.cyan(imageFileNames.length)} images found. ${chalk.cyan(
      paramFileNames.length
    )} params found.`
  );

  return { imageFileNames, paramFileNames };
};

/* ---------------------- List Images Found In Other Folders --------------------- */
export const listImagesFoundInOtherFolders = ({
  imageFileNames,
  imagesInOtherFolders,
  withConsole = false,
}: ImageFileNames & { imagesInOtherFolders: Image[]; withConsole?: boolean }) => {
  const images = imageFileNames.reduce((acc, cur) => {
    const imageInOtherFolder = imagesInOtherFolders.find((img) => img.name === cur);
    if (!imageInOtherFolder) return acc;
    else acc.push(imageInOtherFolder);
    return acc;
  }, [] as Image[]);

  if (withConsole)
    console.log(
      `Images in current folder and other folders:\n${chalk.cyan(
        makeConsoleList(
          images.map((img) => `Name: ${img.name}. Hash: ${img.hashMD5}. Path: ${img.path}.`)
        )
      )}`
    );

  return images;
};

/* ---------------------- List Models ---------------------- */
export const listModels = async (withConsole = false) => {
  const models = (await axios({ method: "GET", url: `${API_URL}/sd-models` }))?.data?.map((m) => ({
    hash: m.hash,
    name: m.model_name.replace(/,|^(\n|\r|\s)|(\n|\r|\s)$/gim, ""),
    path: m.filename,
  })) as { hash: string; name: string; path: string }[];
  if (withConsole)
    console.log(`Models:\n${chalk.cyan(makeConsoleList(models.map((m) => m.name)))}`);
  return models;
};

/* ---------------------- List Samplers ---------------------- */
export const listSamplers = async (withConsole = false) => {
  const samplers = (await axios({ method: "GET", url: `${API_URL}/samplers` }))?.data?.map(
    (s) => s.name
  ) as string[];
  if (withConsole) console.log(`Samplers:\n${chalk.cyan(makeConsoleList(samplers))}`);
  return samplers;
};

/* ---------------------- List Upscalers ---------------------- */
export const listUpscalers = async (withConsole = false) => {
  const upscalers = (await axios({ method: "GET", url: `${API_URL}/upscalers` }))?.data?.map(
    (s) => s.name
  ) as string[];
  if (withConsole) console.log(`Upscalers:\n${chalk.cyan(makeConsoleList(upscalers))}`);
  return upscalers;
};

/* -------------------------------- List VAEs ------------------------------- */
/**
 * Uses hashing algorithm used internally by Automatic1111.
 * @see [Line 138 of /modules/sd_models in related commit](https://github.dev/liamkerr/stable-diffusion-webui/blob/66cad3ab6f1a06a15b7302d1788a1574dd08ce86/modules/sd_models.py#L132)
 */
export const listVAEs = async (withConsole = false) => {
  const vaes = await Promise.all(
    (
      await dirToFilePaths(VAE_DIR)
    )
      .filter((filePath) => /ckpt|safetensors/im.test(filePath))
      .map(async (filePath) => ({
        fileName: path.basename(filePath),
        hash: await sha256File(filePath, { length: 10, offset: 0x100000 }),
      }))
  );

  if (withConsole)
    console.log(
      `VAEs:\n${chalk.cyan(
        makeConsoleList(
          vaes.map(
            (v) =>
              `${chalk.blueBright("Filename:")} ${v.fileName}. ${chalk.blueBright("Hash:")} ${
                v.hash
              }.`
          )
        )
      )}`
    );
  return vaes;
};

/* --------------------- Load Txt2Img Generation Overrides From File --------------------- */
export const loadTxt2ImgOverrides = async (paramFileName: string) => {
  try {
    const dirname = path.dirname(`${paramFileName}.txt`);
    const res = await fs.readFile(path.resolve(dirname, TXT2IMG_OVERRIDES_FILE_NAME), {
      encoding: "utf8",
    });
    return JSON.parse(res) as Txt2ImgOverrides;
  } catch (err) {
    return {};
  }
};

/* --------------------- Load Lora Training Params From File --------------------- */
export const loadLoraTrainingParams = async (paramFilePath: string) => {
  try {
    const res = await fs.readFile(paramFilePath, { encoding: "utf8" });
    return JSON.parse(res) as LoraTrainingParams;
  } catch (err) {
    return {};
  }
};

/* -------------------------- Make Time Console Log ------------------------- */
export const makeTimeLog = (time: number) =>
  `${chalk.green(dayjs.duration(time).format("H[h]m[m]s[s]"))} ${chalk.grey(`(${round(time)}ms)`)}`;

/* -------------------- Parse Image Generation Parameter -------------------- */
export const parseImageParam = <IsNum extends boolean>(
  imageParams: string,
  paramName: string,
  isNumber: IsNum,
  optional = false,
  endDelimiter = ",",
  startDelimeter = ": "
): IsNum extends true ? number : string => {
  try {
    const hasParam = imageParams.includes(`${paramName}: `);
    if (!hasParam) {
      if (!optional)
        throw new Error(`Param "${paramName}" not found in image params: ${imageParams}.`);
      return undefined;
    }

    const rawParamUnterminated = imageParams.substring(
      imageParams.indexOf(`${paramName}${startDelimeter}`)
    );
    const startIndex = rawParamUnterminated.indexOf(startDelimeter) + startDelimeter.length;
    let endIndex = rawParamUnterminated.indexOf(endDelimiter, startIndex);
    if (!(endIndex > 0)) endIndex = undefined;

    const value = rawParamUnterminated
      .substring(startIndex, endIndex)
      ?.replace?.(/^(\s|\r)|(\s|\r)$/gim, "");
    if (isNumber) {
      if (isNaN(+value)) throw new Error(`Received NaN when parsing ${paramName}`);
      return +value as any;
    } else return value as any;
  } catch (err) {
    console.error(chalk.red(err));
    return undefined;
  }
};

/* ----------------- Parse Image Generation Parameters File ----------------- */
export const parseImageParams = async ({
  models,
  paramFileName,
}: {
  models: Model[];
  paramFileName: string;
}) => {
  const [imageParams, overrides] = await Promise.all([
    await fs.readFile(`${paramFileName}.txt`, { encoding: "utf8" }),
    await loadTxt2ImgOverrides(paramFileName),
  ]);

  const negPromptEndIndex = imageParams.indexOf("Steps: ");
  let negPromptStartIndex = imageParams.indexOf("Negative prompt: ");
  if (negPromptStartIndex < 0) negPromptStartIndex = negPromptEndIndex;

  const prompt = imageParams.substring(0, negPromptStartIndex).replace(/(\n|\r)$/gim, "");
  const negPrompt = imageParams
    .substring(negPromptStartIndex, negPromptEndIndex)
    .replace(/(\n|\r)|Negative prompt:\s/gim, "");
  const restParams = imageParams.substring(negPromptEndIndex);

  const rawModelName = overrides?.model ?? parseImageParam(restParams, "Model", false);
  const modelHash = parseImageParam(restParams, "Model hash", false);

  let model = remapModelName(models, modelHash);
  if (!model) {
    model = rawModelName;
    console.log(
      chalk.yellow(
        `Invalid model name ${chalk.yellow(rawModelName)} found for ${chalk.cyan(paramFileName)}.`
      )
    );
  }

  /* ------------------------------ Main Settings ----------------------------- */
  const cfgScale = overrides?.cfgScale ?? parseImageParam(restParams, "CFG scale", true);
  const clipSkip = overrides?.clipSkip ?? parseImageParam(restParams, "Clip skip", true, true);
  const hasRestoreFaces = overrides?.restoreFaces ?? !!restParams.includes("Face restoration");
  const hiresDenoisingStrength = overrides?.denoisingStrength ?? DEFAULT_HIRES_DENOISING_STRENGTH;
  const hiresScale = overrides?.hiresScale ?? DEFAULT_HIRES_SCALE;
  const hiresUpscaler = overrides?.upscaler ?? DEFAULT_HIRES_UPSCALER;
  const sampler = overrides?.sampler ?? parseImageParam(restParams, "Sampler", false);
  const seed = overrides?.seed ?? parseImageParam(restParams, "Seed", true);
  const steps = overrides?.steps ?? parseImageParam(restParams, "Steps", true);
  const subseed = overrides?.subseed ?? parseImageParam(restParams, "Variation seed", true, true);
  const subseedStrength =
    overrides?.subseedStrength ??
    parseImageParam(restParams, "Variation seed strength", true, true);
  const vaeHash =
    overrides?.vae ?? parseImageParam(restParams, '"vae"', false, true, '"', ': "') ?? "None";
  const [width, height] = parseImageParam(restParams, "Size", false)
    .split("x")
    .map((d) => +d);

  /* --------------------------- "Cutoff" Extension --------------------------- */
  const cutoffEnabled =
    overrides?.cutoffEnabled ??
    parseImageParam(restParams, "Cutoff enabled", false, true) === "True";
  const cutoffTargets = cutoffEnabled
    ? overrides?.cutoffTargets ??
      (JSON.parse(
        parseImageParam(
          restParams.replace(/\"/gm, ""),
          "Cutoff targets",
          false,
          true,
          "],"
        ).replace(/\'/gm, '"') + "]"
      ) as string[])
    : undefined;
  const cutoffPadding =
    overrides?.cutoffPadding ?? parseImageParam(restParams, "Cutoff padding", false, true);
  const cutoffWeight =
    overrides?.cutoffWeight ?? parseImageParam(restParams, "Cutoff weight", true, true);
  const cutoffDisableForNeg =
    overrides?.cutoffDisableForNeg ??
    parseImageParam(restParams, "Cutoff disable_for_neg", false, true) === "True";
  const cutoffStrong =
    overrides?.cutoffStrong ?? parseImageParam(restParams, "Cutoff strong", false, true) === "True";
  const cutoffInterpolation =
    overrides?.cutoffInterpolation ??
    parseImageParam(restParams, "Cutoff interpolation", false, true);

  /* ----------------------- "Dynamic Prompts" Extension ---------------------- */
  const template =
    overrides?.template ??
    parseImageParam(restParams, "Template", false, true, "Negative Template");
  const negTemplate =
    overrides?.negTemplate ?? parseImageParam(restParams, "Negative Template", false, true, "\r");

  return {
    cfgScale,
    clipSkip,
    cutoffEnabled,
    cutoffTargets,
    cutoffPadding,
    cutoffWeight,
    cutoffDisableForNeg,
    cutoffStrong,
    cutoffInterpolation,
    hasRestoreFaces,
    height,
    hiresDenoisingStrength,
    hiresScale,
    hiresUpscaler,
    model,
    negPrompt,
    negTemplate,
    paramFileName,
    prompt,
    rawParams: imageParams,
    sampler,
    seed,
    steps,
    subseed,
    subseedStrength,
    template,
    width,
    vaeHash,
  };
};

/* -------------- Parse and Sort Image Parameter Files by Model ------------- */
export const parseAndSortImageParams = async ({ imageFileNames, paramFileNames }: FileNames) => {
  if (paramFileNames.length > imageFileNames.length) {
    console.log(
      `Pruning ${chalk.yellow(
        paramFileNames.length - imageFileNames.length
      )} unused generation parameters...`
    );
    await pruneImageParams({ imageFileNames, paramFileNames });
  }

  console.log("Parsing and sorting image params...");
  const models = await listModels();
  const allImageParams = (
    await Promise.all(
      paramFileNames.map((paramFileName) => parseImageParams({ models, paramFileName }))
    )
  ).sort((a, b) => {
    if (a.vaeHash && !b.vaeHash) return -1;
    if (!a.vaeHash && b.vaeHash) return 1;
    if (a.vaeHash && b.vaeHash) return a.vaeHash.localeCompare(b.vaeHash);
    return a.model.localeCompare(b.model);
  });

  console.log(chalk.green("Image params parsed and sorted."));
  return allImageParams;
};

/* ----------------------- Prompt For Batch Generation Parameters ---------------------- */
export const promptForBatchParams = async ({ imageFileNames }: ImageFileNames) => {
  const folderName = await prompt(chalk.blueBright(`Enter folder name: `));

  let excludedPaths = delimit(
    await prompt(
      chalk.blueBright(
        `Enter folder paths to exclude (delineated by |) ${chalk.grey(
          `(${DEFAULT_BATCH_EXCLUDED_PATHS.join(" | ")})`
        )}: `
      )
    ),
    "|"
  );

  if (!excludedPaths.length) excludedPaths = DEFAULT_BATCH_EXCLUDED_PATHS;

  const imagesInOtherFolders =
    excludedPaths.length > 0 ? await getImagesInFolders({ paths: excludedPaths }) : null;
  const imagesInCurFolderAndOtherFolders =
    excludedPaths.length > 0
      ? listImagesFoundInOtherFolders({
          imageFileNames,
          imagesInOtherFolders,
        })
      : null;

  const batchSize =
    +(await prompt(
      `${chalk.blueBright("Enter batch size")} ${chalk.grey(`(${DEFAULT_BATCH_SIZE})`)}:`
    )) || DEFAULT_BATCH_SIZE;

  const filesNotInOtherFolders =
    excludedPaths.length > 0
      ? imageFileNames.filter(
          (imageFileName) =>
            !imagesInCurFolderAndOtherFolders.find((img) => img.name === imageFileName)
        )
      : imageFileNames;

  return { batchSize, filesNotInOtherFolders, folderName };
};

/* ---------------------------- Remap Model Name ---------------------------- */
export const remapModelName = (models: { hash: string; name: string }[], modelHash: string) =>
  models.find((m) => m.hash === modelHash)?.name;

/* ----------------------------- Refresh Models ----------------------------- */
export const refreshModels = async () => {
  console.log("Refreshing models...");
  await axios({ method: "POST", url: `${API_URL}/refresh-checkpoints` });
  console.log(chalk.green("Models refreshed."));
};

/* ---------------------------- Set Active Model ---------------------------- */
export const setActiveModel = async (modelName: string) => {
  const res = await axios({
    method: "POST",
    url: `${API_URL}/options`,
    data: { sd_model_checkpoint: modelName },
  });

  if (res.status !== 200) throw new Error(`Failed to set active model to ${modelName}.`);
};

/* ---------------------------- Set Active VAE ---------------------------- */
/** Current bug with setting VAE that requires switching to another VAE or checkpoint before switching to desired VAE, otherwise will not take effect. */
export const setActiveVAE = async (vaeName: "None" | "Automatic" | string) => {
  let res = await axios({
    method: "POST",
    url: `${API_URL}/options`,
    data: { sd_vae: vaeName },
  });
  if (res.status !== 200) throw new Error(`Failed to set active VAE to ${vaeName}.`);

  res = await axios({
    method: "POST",
    url: `${API_URL}/options`,
    data: { sd_vae: "None" },
  });
  if (res.status !== 200) throw new Error(`Failed to set active VAE to ${vaeName}.`);

  res = await axios({
    method: "POST",
    url: `${API_URL}/options`,
    data: { sd_vae: vaeName },
  });
  if (res.status !== 200) throw new Error(`Failed to set active VAE to ${vaeName}.`);
};

/* -------------------------------------------------------------------------- */
/*                                  ENDPOINTS                                 */
/* -------------------------------------------------------------------------- */
/* --------------- Convert Images in Current Directory to JPG --------------- */
export const convertImagesInCurDirToJPG = async () => {
  const nonJPG = (await fs.readdir(".")).filter(
    (fileName) => ![".jpg", ".txt"].includes(path.extname(fileName))
  );
  await convertImagesToJPG(nonJPG);

  await fs.mkdir(DIR_NAMES.nonJPG);
  await Promise.all(
    nonJPG.map((fileName) => fs.rename(fileName, path.join(DIR_NAMES.nonJPG, fileName)))
  );

  console.log(chalk.green(`Converted ${chalk.cyan(nonJPG.length)} files to JPG.`));
  mainEmitter.emit("done");
};

/* ----------------------------- Generate Batch ----------------------------- */
export const generateBatch = async ({ imageFileNames, noEmit }: ImageFileNames & NoEmit) => {
  const targetPath = (await prompt(chalk.blueBright("Enter target path: "))).trim();
  await fs.mkdir(targetPath, { recursive: true });

  const { batchSize, filesNotInOtherFolders } = await promptForBatchParams({ imageFileNames });
  const filesToCopy = filesNotInOtherFolders.slice(0, batchSize);

  await Promise.all(
    filesToCopy
      .map(async (imageFileName) => {
        const fileName = `${imageFileName}.jpg`;

        fs.copyFile(fileName, path.join(targetPath, fileName)).catch((err) =>
          console.error(chalk.red(err))
        );

        console.log(`Moved ${chalk.cyan(imageFileName)} to ${chalk.magenta(targetPath)}.`);
      })
      .flat()
  );

  console.log(
    chalk.green(`New batch of size ${chalk.cyan(batchSize)} created at ${chalk.cyan(targetPath)}.`)
  );
  if (!noEmit) mainEmitter.emit("done");
};

/* ----------------------------- Generate Batches (Exhaustive) ----------------------------- */
export const generateBatchesExhaustive = async ({
  imageFileNames,
  shuffle = false,
  noEmit,
}: ImageFileNames & NoEmit & { shuffle?: boolean }) => {
  const { batchSize, filesNotInOtherFolders, folderName } = await promptForBatchParams({
    imageFileNames,
  });

  const batches = chunkArray(
    shuffle ? randomSort(filesNotInOtherFolders) : filesNotInOtherFolders,
    batchSize
  ).map((batch, i) => ({
    fileNames: batch,
    folderName: `[Batch] ${folderName} - ${i + 1} (${batch.length})`,
  }));

  await Promise.all(
    batches.map(async (batch) => {
      await fs.mkdir(batch.folderName, { recursive: true });

      await Promise.all(
        batch.fileNames.map(async (name) => {
          try {
            const fileName = `${name}.jpg`;
            await fs.copyFile(fileName, path.join(batch.folderName, path.basename(fileName)));
            console.log(`Moved ${chalk.cyan(name)} to ${chalk.magenta(batch.folderName)}.`);
          } catch (err) {
            console.error(chalk.red(err));
          }
        })
      );

      console.log(
        chalk.green(
          `New batch of size ${chalk.cyan(batch.fileNames.length)} created at ${chalk.cyan(
            batch.folderName
          )}.`
        )
      );
    })
  );

  if (!noEmit) mainEmitter.emit("done");
};

/* ------------------------- Generate Images (Hires Fix / Segment by Reproducible) ------------------------ */
export const GenQueue = new GenerationQueue();

export const generateImages = async ({
  mode,
  imageFileNames,
  paramFileNames,
}: FileNames & { mode: "reproduce" | "upscale" }) => {
  try {
    const perfStart = performance.now();

    await refreshModels();
    const [activeModel, activeVAEHash, allImageParams, modelNames, vaes] = await Promise.all([
      getActiveModel(),
      getActiveVAE(),
      parseAndSortImageParams({ imageFileNames, paramFileNames }),
      listModels(),
      listVAEs(),
    ]);

    GenQueue.setActiveModel(activeModel);
    GenQueue.setActiveVAEHash(activeVAEHash);

    let completedCount = 0;
    const totalCount = allImageParams.length;

    allImageParams.forEach((imageParams) =>
      GenQueue.add(async () => {
        try {
          if (!imageFileNames.find((fileName) => fileName === imageParams.paramFileName)) {
            throw new Error(
              `Image for ${chalk.yellow(imageParams.paramFileName)} does not exist. Skipping.`
            );
          }

          const iterationPerfStart = performance.now();

          const requestBody = {
            alwayson_scripts: {
              Cutoff: imageParams.cutoffEnabled
                ? {
                    args: [
                      imageParams.cutoffEnabled,
                      imageParams.cutoffTargets.join(", "),
                      imageParams.cutoffWeight,
                      imageParams.cutoffDisableForNeg,
                      imageParams.cutoffStrong,
                      imageParams.cutoffPadding,
                      imageParams.cutoffInterpolation,
                      false, // debug output
                    ],
                  }
                : undefined,
            },
            cfg_scale: imageParams.cfgScale,
            denoising_strength: imageParams.hiresDenoisingStrength,
            enable_hr: mode === "upscale",
            height: imageParams.height,
            hr_scale: imageParams.hiresScale,
            hr_upscaler: imageParams.hiresUpscaler,
            negative_prompt: imageParams.negPrompt,
            override_settings: { CLIP_stop_at_last_layers: imageParams.clipSkip ?? 1 },
            override_settings_restore_afterwards: true,
            prompt: imageParams.prompt,
            restore_faces: imageParams.hasRestoreFaces,
            sampler_name: imageParams.sampler,
            save_images: false,
            seed: imageParams.seed,
            send_images: true,
            steps: imageParams.steps,
            subseed: imageParams.subseed ?? -1,
            subseed_strength: imageParams.subseedStrength ?? 0,
            width: imageParams.width,
          };

          console.log(
            `${mode === "upscale" ? "Upscaling" : "Reproducing"} ${chalk.cyan(
              imageParams.paramFileName
            )} with params:`,
            chalk.cyan(JSON.stringify(requestBody, null, 2))
          );

          if (imageParams.model !== GenQueue.getActiveModel()) {
            if (!modelNames.map((m) => m.name).includes(imageParams.model)) {
              throw new Error(
                `Model ${chalk.magenta(imageParams.model)} does not exist. Skipping ${
                  imageParams.paramFileName
                }.`
              );
            }

            console.log(`Setting active model to ${chalk.magenta(imageParams.model)}...`);
            await setActiveModel(imageParams.model);
            GenQueue.setActiveModel(imageParams.model);

            console.log(chalk.green("Active model updated."));
          }

          const activeVAEHash = GenQueue.getActiveVAEHash();
          if (imageParams.vaeHash !== activeVAEHash) {
            if (
              ["Automatic", "None"].includes(imageParams.vaeHash) &&
              !["Automatic", "None"].includes(activeVAEHash)
            ) {
              console.log(`Setting active VAE to ${chalk.magenta(imageParams.vaeHash)}...`);
              await setActiveVAE(imageParams.vaeHash);
              GenQueue.setActiveVAEHash(imageParams.vaeHash);
            } else {
              const vae = vaes.find((v) => v.hash === imageParams.vaeHash);
              if (!vae) {
                console.warn(
                  chalk.yellow(
                    `VAE with hash ${chalk.magenta(
                      imageParams.vaeHash
                    )} does not exist. Setting active VAE to ${chalk.magenta("None")}...`
                  )
                );

                await setActiveVAE("None");
                GenQueue.setActiveVAEHash("None");
              } else {
                console.log(
                  `Setting active VAE to ${chalk.magenta(`${vae.fileName} (${vae.hash})`)}...`
                );
                await setActiveVAE(vae.fileName);
                GenQueue.setActiveVAEHash(vae.hash);
              }
            }

            console.log(chalk.green("Active VAE updated."));
          }

          console.log(`Beginning ${mode === "upscale" ? "upscale" : "reproduction"}...`);

          const res = (
            await axios({
              method: "POST",
              url: `${API_URL}/txt2img`,
              headers: { "Content-Type": "application/json" },
              data: requestBody,
            })
          )?.data;

          const genImageBuffer = Buffer.from(res.images[0], "base64");
          const genParams =
            (JSON.parse(res.info).infotexts[0] as string) +
            (imageParams.template ? `\nTemplate: ${imageParams.template}` : "") +
            (imageParams.negTemplate ? `\nNegative Template: ${imageParams.negTemplate}` : "");

          const [imageFileName, paramFileName] = ["jpg", "txt"].map(
            (ext) => `${imageParams.paramFileName}.${ext}`
          );

          if (mode === "reproduce") {
            const { percentDiff, pixelDiff } = await compareImage(imageFileName, genImageBuffer);
            const isReproducible = percentDiff < REPROD_DIFF_TOLERANCE;
            const chalkColor = isReproducible ? "green" : "yellow";
            console.log(
              `Pixel Diff: ${chalk[chalkColor](pixelDiff)}. Percent diff: ${
                chalk[chalkColor](round(percentDiff * 100)) + "%"
              }.`
            );

            const parentDir = path.join(
              path.dirname(imageFileName),
              isReproducible ? DIR_NAMES.reproducible : DIR_NAMES.nonReproducible
            );
            const productsDir = path.join(parentDir, DIR_NAMES.products);
            await fs.mkdir(productsDir, { recursive: true });

            await Promise.all([
              fs.writeFile(path.join(productsDir, path.basename(imageFileName)), genImageBuffer),
              fs.writeFile(path.join(productsDir, path.basename(paramFileName)), genParams),
            ]);

            await Promise.all(
              [imageFileName, paramFileName].map((fileName) =>
                fs.rename(fileName, path.join(parentDir, path.basename(fileName)))
              )
            );

            console.log(
              `${chalk.cyan(imageParams.paramFileName)} moved to ${chalk[
                isReproducible ? "green" : "yellow"
              ](parentDir)}.`
            );
          } else {
            const outputDir = DIR_NAMES.upscaled;
            const sourcesDir = DIR_NAMES.upscaleCompleted;

            await Promise.all(
              [outputDir, sourcesDir].map((dir) => fs.mkdir(dir, { recursive: true }))
            );

            await Promise.all([
              fs.writeFile(path.join(outputDir, path.basename(imageFileName)), genImageBuffer),
              fs.writeFile(path.join(outputDir, path.basename(paramFileName)), genParams),
            ]);

            await Promise.all(
              [imageFileName, paramFileName].map((fileName) =>
                fs.rename(fileName, path.join(sourcesDir, path.basename(fileName)))
              )
            );
          }

          completedCount++;
          const isComplete = completedCount === totalCount;

          const timeElapsed = performance.now() - iterationPerfStart;
          console.log(
            `${chalk.cyan(completedCount)} / ${chalk.grey(totalCount)} completed in ${makeTimeLog(
              timeElapsed
            )}.\n${chalk.grey("-".repeat(100))}`
          );

          if (isComplete) {
            const totalTimeElapsed = performance.now() - perfStart;
            console.log(
              `All ${chalk.cyan(totalCount)} images completed in ${makeTimeLog(totalTimeElapsed)}.`
            );
            mainEmitter.emit("done");
          }
        } catch (err) {
          console.error(chalk.red(err.stack));
        }
      })
    );
  } catch (err) {
    console.error(chalk.red(err.stack));
  }
};

/* ---------------- Generate Lora Training Folder and Params ---------------- */
export const generateLoraTrainingFolderAndParams = async () => {
  const trainingParams = await loadLoraTrainingParams(DEFAULT_LORA_PARAMS_PATH);

  const curDirName = path.basename(await fs.realpath("."));
  const loraName =
    (await prompt(`Enter Lora name ${chalk.grey(`(${curDirName})`)}: `)) || curDirName;
  trainingParams.output_name = loraName;

  const modelList = await listModels(true);
  const modelIndex = +(await prompt(
    `${chalk.blueBright(
      `Enter model to train on (1 - ${modelList.length}) ${chalk.grey(`(${DEFAULT_LORA_MODEL})`)}:`
    )}\n${makeConsoleList(
      modelList.map((m) => m.name),
      true
    )}\n`
  ));
  trainingParams.pretrained_model_name_or_path =
    modelIndex > 0 && modelIndex <= modelList.length
      ? modelList[modelIndex - 1].path
      : DEFAULT_LORA_MODEL;

  const { imageFileNames } = await listImageAndParamFileNames();

  const imagesDirName = `input\\${Math.max(100, 1500 / imageFileNames.length)}_${loraName}`;
  await Promise.all(
    [imagesDirName, "logs", "output"].map((dirName) => fs.mkdir(dirName, { recursive: true }))
  );

  trainingParams.logging_dir = path.resolve("logs");
  trainingParams.output_dir = path.resolve("output");
  trainingParams.train_data_dir = path.resolve("input");

  await Promise.all(
    imageFileNames.map((fileName) =>
      fs.rename(`${fileName}.jpg`, path.join(imagesDirName, `${fileName}.jpg`))
    )
  );
  console.log(
    `Moved ${chalk.cyan(imageFileNames.length)} images to ${chalk.magenta(imagesDirName)}.`
  );

  const trainingParamsJson = JSON.stringify(trainingParams, null, 2);
  await fs.writeFile(LORA_TRAINING_PARAMS_FILE_NAME, trainingParamsJson);

  console.log(
    chalk.green(`Created ${LORA_TRAINING_PARAMS_FILE_NAME}:`),
    chalk.cyan(trainingParamsJson)
  );
  mainEmitter.emit("done");
};

/* ---------------------- Generate Txt2Img Overrides ---------------------- */
export const generateTxt2ImgOverrides = async () => {
  const overrides: Txt2ImgOverrides = {};

  const booleanPrompt = async (question: string, optName: string) => {
    const res = (await prompt(chalk.blueBright(`${question} (y/n): `))) as "y" | "n";
    if (res === "y" || res === "n") overrides[optName] = res === "y";
  };

  const numListPrompt = async (label: string, optName: string, options: string[]) => {
    const index = +(await prompt(
      `${chalk.blueBright(`Enter ${label} (1 - ${options.length}):`)}\n${makeConsoleList(
        options,
        true
      )}\n`
    ));

    if (index > 0 && index <= options.length) overrides[optName] = options[index - 1];
  };

  const numericalPrompt = async (question: string, optName: string) => {
    const res = await prompt(chalk.blueBright(question));
    if (res.length > 0 && !isNaN(+res)) overrides[optName] = +res;
  };

  await numericalPrompt("Enter CFG Scale: ", "cfgScale");
  await numericalPrompt("Enter Clip Skip: ", "clipSkip");
  await numericalPrompt("Enter Denoising Strength: ", "denoisingStrength");
  await numericalPrompt("Enter Hires Scale: ", "hiresScale");
  const models = (await listModels()).map((m) => m.name);
  await numListPrompt("Model", "model", models);
  await booleanPrompt("Restore Faces?", "restoreFaces");
  const samplers = await listSamplers();
  await numListPrompt("Sampler", "sampler", samplers);
  await numericalPrompt("Enter Seed: ", "seed");
  await numericalPrompt("Enter Steps: ", "steps");
  await numericalPrompt("Enter Subseed: ", "subseed");
  await numericalPrompt("Enter Subseed Strength: ", "subseedStrength");
  const upscalers = await listUpscalers();
  await numListPrompt("Upscaler", "upscaler", upscalers);
  const vaes = await listVAEs();
  await numListPrompt("VAE", "vae", ["None", "Automatic"].concat(vaes.map((v) => v.fileName)));

  const overridesJson = JSON.stringify(overrides, null, 2);
  await fs.writeFile(TXT2IMG_OVERRIDES_FILE_NAME, overridesJson);

  console.log(chalk.green(`Created ${TXT2IMG_OVERRIDES_FILE_NAME}:`), chalk.cyan(overridesJson));
  mainEmitter.emit("done");
};

/* --------------- Initialize Automatic1111 Folders (Symlinks) -------------- */
export const initAutomatic1111Folders = async () => {
  const promptForPath = async (question: string) =>
    (await prompt(chalk.blueBright(question))).replace(/\"|^(\r|\s)|(\r|\s)$/gim, "");

  try {
    const rootDir = await fs.realpath(".");
    console.log("Root Automatic1111 folder path: ", chalk.cyan(rootDir));

    const promptForSymlink = async (targetDirName: string, sourceDir: string) => {
      const folder = await promptForPath(
        `Enter path of external ${chalk.magenta(targetDirName)} folder: `
      );
      if (!folder.length)
        console.warn(chalk.yellow(`Not linking ${chalk.magenta(targetDirName)} folder.`));
      else {
        const hasSourceDir = await doesPathExist(sourceDir);
        if (hasSourceDir) await fs.rename(sourceDir, `${sourceDir} (OLD)`);

        await fs.symlink(folder, path.join(rootDir, sourceDir), "junction");
        console.log(chalk.green(`${targetDirName} folder linked.`));
      }
    };

    await promptForSymlink("Embeddings", "embeddings");
    await promptForSymlink("Extensions", "extensions");
    await promptForSymlink("Lora", "models\\Lora");
    await promptForSymlink("LyCORIS / LoCon / LoHa", "models\\LyCORIS");
    await promptForSymlink("Models", "models\\Stable-diffusion");
    await promptForSymlink("Outputs", "outputs");
    await promptForSymlink("VAE", "models\\VAE");
    await promptForSymlink("ControlNet Poses", "models\\ControlNet");
  } catch (err) {
    console.error(chalk.red(err));
  } finally {
    mainEmitter.emit("done");
  }
};

/* ------------------- Prune Files Found in Other Folders ------------------- */
export const pruneFilesFoundInFolders = async ({
  imageFileNames,
  noEmit,
}: ImageFileNames & NoEmit) => {
  const otherPaths = (await prompt(chalk.blueBright("Enter folder paths to exclude: ")))
    .replace(/^(,|\s)|(,|\s)$/g, "")
    .split("|")
    .map((p) => p.trim());

  const imagesInOtherFolders = await getImagesInFolders({ paths: otherPaths });

  const imagesInCurFolderAndOtherFolders = listImagesFoundInOtherFolders({
    imageFileNames,
    imagesInOtherFolders,
  });

  const targetDir = DIR_NAMES.prunedImagesOtherFolders;
  await fs.mkdir(targetDir, { recursive: true });

  await Promise.all(
    imagesInCurFolderAndOtherFolders.map(async (img) => {
      const fileName = `${img.name}.${img.ext}`;
      await fs.rename(fileName, path.join(targetDir, fileName));
      console.log(`Pruned ${chalk.yellow(fileName)}.`);
    })
  );

  console.log(chalk.green("All images found in other folders pruned."));
  if (!noEmit) mainEmitter.emit("done");
};

/* ------------------- Prune Generation Parameters ------------------- */
export const pruneImageParams = async ({
  noEmit,
  imageFileNames,
  paramFileNames,
}: FileNames & NoEmit) => {
  const unusedParams: string[] = [];
  const dirName = DIR_NAMES.prunedParams;
  await fs.mkdir(dirName, { recursive: true });

  await Promise.all(
    paramFileNames.map(async (p) => {
      const image = imageFileNames.find((i) => i === p);
      if (!image) {
        const fileName = `${p}.txt`;
        await fs.rename(fileName, path.join(dirName, path.basename(fileName)));
        unusedParams.push(p);
        console.log(chalk.yellow(`${fileName} pruned.`));
      }
    })
  );

  console.log(`${chalk.yellow(unusedParams.length)} unused params pruned.`);
  if (!noEmit) mainEmitter.emit("done");

  return unusedParams;
};

/* ------------ Prune Generation Parameters & Segment by Upscaled ----------- */
export const pruneParamsAndSegmentUpscaled = async (fileNames: FileNames) => {
  try {
    const prunedParams = await pruneImageParams({ ...fileNames, noEmit: true });
    const filteredParamNames = fileNames.paramFileNames.filter((p) => !prunedParams.includes(p));
    await segmentByUpscaled({ paramFileNames: filteredParamNames, noEmit: true });
  } catch (err) {
    console.error(chalk.red(err));
  } finally {
    mainEmitter.emit("done");
  }
};

/* -------------------------- Segment by Dimensions ------------------------- */
export const segmentByDimensions = async ({ imageFileNames, noEmit }: ImageFileNames & NoEmit) => {
  console.log("Segmenting by dimensions...");

  await Promise.all(
    imageFileNames.map(async (fileName) => {
      const { height, width } = await sharp(`${fileName}.jpg`).metadata();
      const dirName = `${height} x ${width}`;

      await fs.mkdir(dirName, { recursive: true });
      await Promise.all(
        ["jpg", "txt"].map((ext) => fs.rename(fileName, path.join(dirName, `${fileName}.${ext}`)))
      );
      console.log(`Moved ${chalk.cyan(fileName)} to ${chalk.magenta(dirName)}.`);
    })
  );

  if (!noEmit) mainEmitter.emit("done");
};

/* --------------------------- Segment by Keywords -------------------------- */
export const segmentByKeywords = async ({
  imageFileNames,
  paramFileNames,
  noEmit,
}: FileNames & NoEmit) => {
  const promptForKeywords = async (type: "all" | "any") =>
    (
      await prompt(
        `Enter keywords - ${type === "all" ? "all" : "at least one"} required ${chalk(
          "(delineated by commas)"
        )}: `
      )
    )
      .replace(/^(,|\s)|(,|\s)$/g, "")
      .split(",")
      .map((p) => p.trim())
      .filter((keyword) => keyword.length);

  const requiredAllKeywords = await promptForKeywords("all");
  const requiredAnyKeywords = await promptForKeywords("any");

  const allImageParams = await parseAndSortImageParams({ imageFileNames, paramFileNames });

  console.log("Segmenting by keyword...");
  let segmentedCount = 0;

  await Promise.all(
    allImageParams.map(async (imageParams) => {
      const hasAllRequired =
        requiredAllKeywords.length > 0
          ? requiredAllKeywords.every((keyword) =>
              new RegExp(escapeRegEx(keyword), "im").test(imageParams.rawParams)
            )
          : true;
      const hasAnyRequired =
        requiredAnyKeywords.length > 0
          ? requiredAnyKeywords.some((keyword) =>
              new RegExp(escapeRegEx(keyword), "im").test(imageParams.rawParams)
            )
          : true;

      if (hasAllRequired && hasAnyRequired) {
        const dirName = `(${requiredAllKeywords.join(" & ")}) [${requiredAnyKeywords.join(" ~ ")}]`;
        await fs.mkdir(dirName, { recursive: true });

        await Promise.all(
          ["jpg", "txt"].map((ext) => {
            const fileName = `${imageParams.paramFileName}.${ext}`;
            return fs.rename(fileName, path.join(dirName, fileName));
          })
        );

        segmentedCount++;
        console.log(`Moved ${chalk.cyan(imageParams.paramFileName)} to ${chalk.magenta(dirName)}.`);
      }
    })
  );

  console.log(chalk.cyan(segmentedCount), chalk.green(" files segmented by keyword."));
  if (!noEmit) mainEmitter.emit("done");
};

/* ---------------------------- Segment by Model ---------------------------- */
export const segmentByModel = async ({
  imageFileNames,
  paramFileNames,
  noEmit,
}: FileNames & NoEmit) => {
  console.log("Segmenting by model...");

  const allImageParams = await parseAndSortImageParams({ imageFileNames, paramFileNames });

  await Promise.all(
    allImageParams.map(async (imageParams) => {
      await fs.mkdir(imageParams.model, { recursive: true });

      await Promise.all(
        ["jpg", "txt"].map((ext) => {
          const fileName = `${imageParams.paramFileName}.${ext}`;
          const newPath = path.join(imageParams.model, fileName);
          return fs.rename(fileName, newPath);
        })
      );

      console.log(
        `Moved ${chalk.cyan(imageParams.paramFileName)} to ${chalk.magenta(imageParams.model)}.`
      );
    })
  );

  console.log(chalk.green("Files segmented by model."));
  if (!noEmit) mainEmitter.emit("done");
};

/* --------------------------- Segment by Upscaled -------------------------- */
export const segmentByUpscaled = async ({ paramFileNames, noEmit }: ParamFileNames & NoEmit) => {
  console.log("Segmenting by upscaled...");

  await Promise.all([
    fs.mkdir(DIR_NAMES.upscaled, { recursive: true }),
    fs.mkdir(DIR_NAMES.nonUpscaled, { recursive: true }),
  ]);

  let upscaledCount = 0;
  let nonUpscaledCount = 0;

  await Promise.all(
    paramFileNames.map(async (i) => {
      const imageParams = await fs.readFile(`${i}.txt`);
      const targetPath = imageParams.includes("Hires upscaler")
        ? DIR_NAMES.upscaled
        : DIR_NAMES.nonUpscaled;
      const isUpscaled = targetPath === DIR_NAMES.upscaled;
      isUpscaled ? upscaledCount++ : nonUpscaledCount++;

      await Promise.all(
        ["jpg", "txt"].map((ext) => fs.rename(`${i}.${ext}`, `${targetPath}\\${i}.${ext}`))
      );
      console.log(`Moved ${i} to ${(isUpscaled ? chalk.green : chalk.yellow)(targetPath)}.`);
    })
  );

  console.log(
    `${chalk.green("Files segmented by upscaled.")} ${chalk.cyan(
      upscaledCount
    )} upscaled images. ${chalk.yellow(nonUpscaledCount)} non-upscaled images.`
  );
  if (!noEmit) mainEmitter.emit("done");
};

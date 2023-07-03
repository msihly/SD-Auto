import fs from "fs/promises";
import EventEmitter from "events";
import path from "path";
import { performance } from "perf_hooks";
import axios, { AxiosRequestConfig } from "axios";
import chalk from "chalk";
import dayjs from "dayjs";
import sharp from "sharp";
import exitHook from "async-exit-hook";
import cliProgress from "cli-progress";
import { main } from "./main";
import {
  API_URL,
  AUTO1111_FOLDER_CONFIG_FILE_NAME,
  DEFAULTS,
  DIR_NAMES,
  EXTS,
  LORA,
  OUTPUT_DIR,
  REPROD_DIFF_TOLERANCE,
  TXT2IMG_OVERRIDES_FILE_NAME,
  VAE_DIR,
} from "./constants";
import {
  Auto1111FolderConfig,
  Auto1111FolderName,
  ExtrasRequest,
  FileNames,
  ImageFileNames,
  ImageParams,
  LoraTrainingParams,
  Model,
  NoEmit,
  ParamFileNames,
  Txt2ImgMode,
  Txt2ImgOverride,
  Txt2ImgOverrideGroup,
  Txt2ImgOverrides,
  Txt2ImgRequest,
  VAE,
} from "./types";
import {
  Image,
  PromiseQueue,
  TreeNode,
  chunkArray,
  compareImage,
  convertImagesToJPG,
  createTree,
  delimit,
  dirToFilePaths,
  doesPathExist,
  escapeRegEx,
  getImagesInFolders,
  makeConsoleList,
  makeExistingValueLog,
  prompt,
  randomSort,
  removeEmptyFolders,
  round,
  sha256File,
  valsToOpts,
} from "./utils";
import env from "./env";

/* -------------------------------------------------------------------------- */
/*                                    UTILS                                   */
/* -------------------------------------------------------------------------- */
const extendFileName = (fileName: string) => ({
  imageFileName: `${fileName}.${EXTS.IMAGE}`,
  paramFileName: `${fileName}.${EXTS.PARAMS}`,
});

/* ------------------------------ Main Emitter ------------------------------ */
const mainEmitter = new EventEmitter();

mainEmitter.on("done", () => {
  setTimeout(() => {
    console.log(chalk.grey("-".repeat(100)));
    main();
  }, 500);
});

/* ------------------------- Automatic1111 REST API ------------------------- */
const apiCall = async (
  method: "GET" | "POST",
  endpoint: string,
  config?: AxiosRequestConfig,
  withThrow = false
) => {
  try {
    const res = await axios({ method, url: `${API_URL}/${endpoint}`, ...config });
    return { success: true, data: res?.data };
  } catch (err) {
    const errMsg = err.message.includes("ECONNREFUSED")
      ? "Automatic1111 server is not connected. Start server to use this command."
      : err.stack;

    if (withThrow) throw new Error(errMsg);
    console.error(chalk.red(`[API Error::${method}] ${endpoint} - ${config}\n${errMsg}`));
    return { success: false, error: errMsg };
  }
};

const API = {
  Auto1111: {
    get: (endpoint: string, config?: AxiosRequestConfig) => apiCall("GET", endpoint, config),
    post: (endpoint: string, config?: AxiosRequestConfig) => apiCall("POST", endpoint, config),
  },
};

/* ---------------------------- Get Active Model ---------------------------- */
export const getActiveModel = async (withConsole = false, withEmit = false) => {
  const res = await API.Auto1111.get("options");
  if (!res.success) return;

  const model = (res.data?.sd_model_checkpoint as string)
    .replace(/\\/gi, "_")
    .replace(/(\.(ckpt|checkpoint|safetensors))|(\[.+\]$)/gi, "")
    .trim();
  if (withConsole) console.log("Active model: ", chalk.cyan(model));
  if (withEmit) mainEmitter.emit("done");
  return model;
};

/* ---------------------------- Get Active VAE ---------------------------- */
export const getActiveVAE = async (withConsole = false, withEmit = false) => {
  const res = await API.Auto1111.get("options");
  if (!res.success) return;

  const vae = (res.data?.sd_vae as string).trim();
  if (withConsole) console.log("Active VAE: ", chalk.cyan(vae));
  if (withEmit) mainEmitter.emit("done");
  return vae;
};

/* -------------------------- Interrupt Generation -------------------------- */
export const interruptGeneration = async (withConsole = false) => {
  if (withConsole) console.log(chalk.yellow("Interrupting generation..."));
  const res = await API.Auto1111.post("interrupt");
  if (withConsole && res.success) console.log(chalk.green("Generation interrupted!"));
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
      const ext = cur.substring(cur.lastIndexOf(".") + 1);
      if (ext === EXTS.IMAGE) acc[0].push(fileName);
      else if (ext === EXTS.PARAMS) acc[1].push(fileName);
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
const listImagesFoundInOtherFolders = ({
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
export const listModels = async (withConsole = false, withEmit = false) => {
  const res = await API.Auto1111.get("sd-models");
  if (!res.success) return;

  const models = res.data?.map((m) => ({
    hash: m.hash,
    name: m.model_name.replace(/,|^(\n|\r|\s)|(\n|\r|\s)$/gim, ""),
    path: m.filename,
  })) as Model[];
  if (withConsole)
    console.log(`Models:\n${chalk.cyan(makeConsoleList(models.map((m) => m.name)))}`);
  if (withEmit) mainEmitter.emit("done");
  return models;
};

/* ---------------------- List Samplers ---------------------- */
export const listSamplers = async (withConsole = false, withEmit = false) => {
  const res = await API.Auto1111.get("samplers");
  if (!res.success) return;

  const samplers = res?.data?.map((s) => s.name) as string[];
  if (withConsole) console.log(`Samplers:\n${chalk.cyan(makeConsoleList(samplers))}`);
  if (withEmit) mainEmitter.emit("done");
  return samplers;
};

/* ---------------------- List Upscalers ---------------------- */
export const listUpscalers = async (withConsole = false, withEmit = false) => {
  const res = await API.Auto1111.get("upscalers");
  if (!res.success) return;

  const upscalers = res?.data?.map((s) => s.name) as string[];
  if (withConsole) console.log(`Upscalers:\n${chalk.cyan(makeConsoleList(upscalers))}`);
  if (withEmit) mainEmitter.emit("done");
  return upscalers;
};

/* -------------------------------- List VAEs ------------------------------- */
/**
 * Uses hashing algorithm used internally by Automatic1111.
 * @see [Line 138 of /modules/sd_models in related commit](https://github.dev/liamkerr/stable-diffusion-webui/blob/66cad3ab6f1a06a15b7302d1788a1574dd08ce86/modules/sd_models.py#L132)
 */
export const listVAEs = async (withConsole = false, withEmit = false) => {
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
  if (withEmit) mainEmitter.emit("done");
  return vaes;
};

/* --------------------- Load Txt2Img Generation Overrides From Files / Folders --------------------- */
const createOverridesTreeNode = async (
  treeNode: TreeNode,
  tree: Txt2ImgOverrideGroup[],
  parentPath: string,
  parentOverrides: Txt2ImgOverrides
) => {
  const fullPath = parentPath ? path.join(parentPath, treeNode.name) : treeNode.name;
  const dirPath = path.extname(fullPath).length > 0 ? path.dirname(fullPath) : fullPath;
  const overrides = { ...parentOverrides, ...(await loadTxt2ImgOverrides(dirPath)) };

  const dirNode = tree.find((t) => t.dirPath === dirPath);
  if (fullPath !== dirPath) {
    const fileName = path.basename(fullPath);
    if (dirNode) dirNode.fileNames.push(fileName);
    else tree.push({ dirPath, overrides, fileNames: [fileName] });
  } else if (!dirNode) tree.push({ dirPath, overrides, fileNames: [] });

  if (treeNode.children.length > 0)
    await Promise.all(
      treeNode.children.map((node) => createOverridesTreeNode(node, tree, fullPath, overrides))
    );
};

const createOverridesTree = async (filePaths: string[]) => {
  const dirTree = createTree(filePaths);

  const overrideTree = [
    { dirPath: ".", fileNames: [], overrides: await loadTxt2ImgOverrides() },
  ] as Txt2ImgOverrideGroup[];

  await Promise.all(
    dirTree.map((treeNode) =>
      createOverridesTreeNode(treeNode, overrideTree, ".", overrideTree[0].overrides)
    )
  );

  return overrideTree.reduce((acc, cur) => {
    if (!cur.fileNames.length) return acc;

    acc.push({
      filePaths: cur.fileNames.map((fileName) => path.join(cur.dirPath, fileName)),
      overrides: cur.overrides,
    });
    return acc;
  }, [] as { filePaths: string[]; overrides: Txt2ImgOverrides }[]);
};

const loadTxt2ImgOverrides = async (dirPath: string = ".") => {
  try {
    const filePath = path.resolve(dirPath, TXT2IMG_OVERRIDES_FILE_NAME);
    if (!(await doesPathExist(filePath))) return {};
    else return JSON.parse(await fs.readFile(filePath, { encoding: "utf8" })) as Txt2ImgOverrides;
  } catch (err) {
    console.error(chalk.red("Error reading txt2img overrides: ", err.stack));
    return {};
  }
};

/* --------------------- Load Auto111 Folder Config --------------------- */
const loadAuto1111FolderConfig = async (dirPath: string = ".") => {
  try {
    const filePath = path.resolve(dirPath, AUTO1111_FOLDER_CONFIG_FILE_NAME);
    if (!(await doesPathExist(filePath))) return {};
    else
      return JSON.parse(await fs.readFile(filePath, { encoding: "utf8" })) as Auto1111FolderConfig;
  } catch (err) {
    console.error(chalk.red("Error reading Automatic1111 folder config: ", err.stack));
    return {};
  }
};

/* --------------------- Load Lora Training Params From File --------------------- */
const loadLoraTrainingParams = async (filePath: string) => {
  try {
    if (!(await doesPathExist(filePath))) return {};
    else return JSON.parse(await fs.readFile(filePath, { encoding: "utf8" })) as LoraTrainingParams;
  } catch (err) {
    console.error(chalk.red("Error reading lora training params: ", err.stack));
    return {};
  }
};

/* -------------------------- Make Time Console Log ------------------------- */
export const makeTimeLog = (time: number) =>
  `${chalk.green(dayjs.duration(time).format("H[h]m[m]s[s]"))} ${chalk.grey(`(${round(time)}ms)`)}`;

/* -------------------- Parse Image Generation Parameter -------------------- */
const parseImageParam = <IsNum extends boolean>(
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
        throw new Error(`Param "${paramName}" not found in generation parameters: ${imageParams}.`);
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
const parseImageParams = async ({
  models,
  overrides,
  paramFileName,
}: {
  models: Model[];
  overrides?: Txt2ImgOverrides;
  paramFileName: string;
}): Promise<ImageParams> => {
  try {
    const fileName = path.join(
      path.dirname(paramFileName),
      path.basename(paramFileName, path.extname(paramFileName))
    );

    const imageParams = await fs.readFile(`${fileName}.${EXTS.PARAMS}`, { encoding: "utf8" });

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
    const hiresDenoisingStrength =
      overrides?.hiresDenoisingStrength ?? DEFAULTS.HIRES_DENOISING_STRENGTH;
    const hiresScale = overrides?.hiresScale ?? DEFAULTS.HIRES_SCALE;
    const hiresSteps =
      overrides?.hiresSteps ?? parseImageParam(restParams, "Hires steps", true, true);
    const hiresUpscaler = overrides?.hiresUpscaler ?? DEFAULTS.HIRES_UPSCALER;
    const restoreFaces = overrides?.restoreFaces ?? !!restParams.includes("Face restoration");
    const restoreFacesStrength = overrides?.restoreFacesStrength ?? DEFAULTS.RESTORE_FACES_STRENGTH;
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
    const parseCutOffTargets = () => {
      const targets =
        parseImageParam(
          restParams.replace(/\"/gm, ""),
          "Cutoff targets",
          false,
          true,
          "],"
        ).replace(/\'/gm, '"') + "]";

      try {
        return JSON.parse(targets) as string[];
      } catch (err) {
        return targets
          .replace(/\[|\]/gim, "")
          .split(",")
          .map((d) => d.trim());
      }
    };

    const cutoffEnabled =
      overrides?.cutoffEnabled ??
      parseImageParam(restParams, "Cutoff enabled", false, true) === "True";
    const cutoffTargets = cutoffEnabled
      ? overrides?.cutoffTargets ?? parseCutOffTargets()
      : undefined;
    const cutoffPadding =
      overrides?.cutoffPadding ?? parseImageParam(restParams, "Cutoff padding", false, true);
    const cutoffWeight =
      overrides?.cutoffWeight ?? parseImageParam(restParams, "Cutoff weight", true, true);
    const cutoffDisableForNeg =
      overrides?.cutoffDisableForNeg ??
      parseImageParam(restParams, "Cutoff disable_for_neg", false, true) === "True";
    const cutoffStrong =
      overrides?.cutoffStrong ??
      parseImageParam(restParams, "Cutoff strong", false, true) === "True";
    const cutoffInterpolation =
      overrides?.cutoffInterpolation ??
      parseImageParam(restParams, "Cutoff interpolation", false, true);

    /* ----------------------- "Dynamic Prompts" Extension ---------------------- */
    const template =
      overrides?.template ??
      parseImageParam(restParams, "Template", false, true, "Negative Template");
    const negTemplate =
      overrides?.negTemplate ?? parseImageParam(restParams, "Negative Template", false, true, "\r");

    /* ----------------- "Multi Diffusion / Tiled VAE" Extension ---------------- */
    const tiledDiffusion =
      overrides.tiledDiffusion ?? env.TILED_DIFFUSION
        ? {
            batchSize: overrides.tiledDiffusion?.batchSize ?? env.TILED_DIFFUSION.batchSize,
            keepInputSize:
              overrides.tiledDiffusion?.keepInputSize ?? env.TILED_DIFFUSION.keepInputSize,
            method: overrides.tiledDiffusion?.method ?? env.TILED_DIFFUSION.method,
            overwriteSize:
              overrides.tiledDiffusion?.overwriteSize ?? env.TILED_DIFFUSION.overwriteSize,
            tileHeight: overrides.tiledDiffusion?.tileHeight ?? env.TILED_DIFFUSION.tileHeight,
            tileOverlap: overrides.tiledDiffusion?.tileOverlap ?? env.TILED_DIFFUSION.tileOverlap,
            tileWidth: overrides.tiledDiffusion?.tileWidth ?? env.TILED_DIFFUSION.tileWidth,
          }
        : undefined;

    const tiledVAE =
      overrides.tiledVAE ?? env.TILED_VAE
        ? {
            colorFixEnabled: overrides.tiledVAE?.colorFixEnabled ?? env.TILED_VAE.colorFixEnabled,
            decoderTileSize: overrides.tiledVAE?.decoderTileSize ?? env.TILED_VAE.decoderTileSize,
            encoderTileSize: overrides.tiledVAE?.encoderTileSize ?? env.TILED_VAE.encoderTileSize,
            fastDecoderEnabled:
              overrides.tiledVAE?.fastDecoderEnabled ?? env.TILED_VAE.fastDecoderEnabled,
            fastEncoderEnabled:
              overrides.tiledVAE?.fastEncoderEnabled ?? env.TILED_VAE.fastEncoderEnabled,
            vaeToGPU: overrides.tiledVAE?.vaeToGPU ?? env.TILED_VAE.vaeToGPU,
          }
        : undefined;

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
      fileName,
      height,
      hiresDenoisingStrength,
      hiresScale,
      hiresSteps,
      hiresUpscaler,
      model,
      negPrompt,
      negTemplate,
      prompt,
      rawParams: imageParams,
      restoreFaces,
      restoreFacesStrength,
      sampler,
      seed,
      steps,
      subseed,
      subseedStrength,
      template,
      tiledDiffusion,
      tiledVAE,
      width,
      vaeHash,
    };
  } catch (err) {
    console.log(chalk.red(`Error parsing ${chalk.cyan(paramFileName)}: ${err.message}`));
    return undefined;
  }
};

/* -------------- Parse and Sort Image Parameter Files by Model ------------- */
const parseAndSortImageParams = async (fileNames: FileNames): Promise<ImageParams[]> => {
  let { imageFileNames, paramFileNames } = fileNames;

  if (paramFileNames.length > imageFileNames.length) {
    console.log(
      `Pruning ${chalk.yellow(
        paramFileNames.length - imageFileNames.length
      )} unused generation parameters...`
    );
    const prunedParams = await pruneImageParams({ imageFileNames, paramFileNames });
    paramFileNames = fileNames.paramFileNames.filter((p) => !prunedParams.includes(p));
  }

  console.log("Parsing and sorting generation parameters...");
  const models = await listModels();

  const overridesTree = await createOverridesTree(paramFileNames.map((f) => `${f}.${EXTS.PARAMS}`));

  const allImageParams = (
    await Promise.all(
      overridesTree.flatMap(({ overrides, filePaths }) =>
        filePaths.map((filePath) =>
          parseImageParams({ models, overrides, paramFileName: filePath })
        )
      )
    )
  )
    .filter((p) => !!p)
    .sort((a, b) => {
      if (a.vaeHash && !b.vaeHash) return -1;
      if (!a.vaeHash && b.vaeHash) return 1;
      if (a.vaeHash && b.vaeHash) return a.vaeHash.localeCompare(b.vaeHash);
      return a.model.localeCompare(b.model);
    });

  console.log(chalk.green("Generation parameters parsed and sorted."));
  return allImageParams;
};

/* ----------------------- Prompt For Batch Generation Parameters ---------------------- */
const promptForBatchParams = async ({ imageFileNames }: ImageFileNames) => {
  const folderName = await prompt(chalk.blueBright(`Enter folder name: `));

  let excludedPaths = delimit(
    await prompt(
      chalk.blueBright(
        `Enter folder paths to exclude (delineated by |) ${chalk.grey(
          `(${DEFAULTS.BATCH_EXCLUDED_PATHS.join(" | ")})`
        )}: `
      )
    ),
    "|"
  );

  if (!excludedPaths.length) excludedPaths = DEFAULTS.BATCH_EXCLUDED_PATHS;

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
      `${chalk.blueBright("Enter batch size")} ${chalk.grey(`(${DEFAULTS.BATCH_SIZE})`)}:`
    )) || DEFAULTS.BATCH_SIZE;

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
const remapModelName = (models: { hash: string; name: string }[], modelHash: string) =>
  models.find((m) => m.hash === modelHash)?.name;

/* ----------------------------- Refresh Models ----------------------------- */
const refreshModels = async () => {
  console.log("Refreshing models...");
  const res = await API.Auto1111.post("refresh-checkpoints");
  if (res.success) console.log(chalk.green("Models refreshed."));
};

/* ---------------------------- Set Active Model ---------------------------- */
const setActiveModel = async (modelName: string) => {
  const res = await API.Auto1111.post("options", { data: { sd_model_checkpoint: modelName } });
  if (!res.success) return;
};

/* ---------------------------- Set Active VAE ---------------------------- */
const setActiveVAE = async (vaeName: "None" | "Automatic" | string) => {
  /** Current bug with setting VAE that requires switching to another VAE or checkpoint before switching to desired VAE, otherwise will not take effect. */
  let res = await API.Auto1111.post("options", { data: { sd_vae: vaeName } });
  if (!res.success) return;
  res = await API.Auto1111.post("options", { data: { sd_vae: "None" } });
  if (!res.success) return;
  res = await API.Auto1111.post("options", { data: { sd_vae: vaeName } });
  if (!res.success) return;
};

/* -------------------------------------------------------------------------- */
/*                                  ENDPOINTS                                 */
/* -------------------------------------------------------------------------- */
/* --------------- Convert Images in Current Directory to JPG --------------- */
export const convertImagesInCurDirToJPG = async () => {
  const nonJPG = (await fs.readdir(".")).filter(
    (fileName) => ![EXTS.IMAGE, EXTS.PARAMS].includes(path.extname(fileName).substring(1))
  );
  await convertImagesToJPG(nonJPG);

  await fs.mkdir(DIR_NAMES.nonJPG);
  await Promise.all(
    nonJPG.map((fileName) => fs.rename(fileName, path.join(DIR_NAMES.nonJPG, fileName)))
  );

  console.log(chalk.green(`Converted ${chalk.cyan(nonJPG.length)} files to JPG.`));
  mainEmitter.emit("done");
};

/* ----------------------------- Generate Batches Exhaustively ----------------------------- */
export const generateBatches = async ({
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
            const fileName = `${name}.${EXTS.IMAGE}`;
            await fs.copyFile(fileName, path.join(batch.folderName, path.basename(fileName)));
            console.log(`Copied ${chalk.cyan(name)} to ${chalk.magenta(batch.folderName)}.`);
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

/* ------------------------- Generate Images (Hires Fix / Reproduce) ------------------------ */
class GenerationQueue extends PromiseQueue {
  activeModel: string = null;

  activeVAEHash: string = null;

  progress = new cliProgress.SingleBar(
    {
      format: `${chalk.cyan("{bar}")} {percentage}% | ${chalk.blueBright(
        "Elapsed:"
      )} {duration_formatted} | ${chalk.blueBright("ETA:")} {formattedETA}`,
    },
    cliProgress.Presets.shades_classic
  );

  progressInterval: NodeJS.Timer = null;

  createTxt2ImgRequest(imageParams: ImageParams, mode: Txt2ImgMode): Txt2ImgRequest {
    return {
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
        "Tiled Diffusion": imageParams.tiledDiffusion
          ? {
              args: [
                "True", // Tiled Diffusion enabled
                imageParams.tiledDiffusion.method, // "MultiDiffusion"
                imageParams.tiledDiffusion.overwriteSize, // "False"
                imageParams.tiledDiffusion.keepInputSize, // "True"
                imageParams.width,
                imageParams.height,
                imageParams.tiledDiffusion.tileWidth, // 128
                imageParams.tiledDiffusion.tileHeight, // 128
                imageParams.tiledDiffusion.tileOverlap, // 48
                imageParams.tiledDiffusion.batchSize, // 4
                imageParams.hiresUpscaler,
                imageParams.hiresScale,
              ],
            }
          : undefined,
        "Tiled VAE": imageParams.tiledVAE
          ? {
              args: [
                "True", // Tiled VAE enabled
                imageParams.tiledVAE.encoderTileSize,
                imageParams.tiledVAE.decoderTileSize,
                imageParams.tiledVAE.vaeToGPU,
                imageParams.tiledVAE.fastDecoderEnabled,
                imageParams.tiledVAE.fastEncoderEnabled,
                imageParams.tiledVAE.colorFixEnabled,
              ],
            }
          : undefined,
      },
      cfg_scale: imageParams.cfgScale,
      denoising_strength: imageParams.hiresDenoisingStrength,
      enable_hr: mode === "upscale",
      height: imageParams.height,
      hr_scale: imageParams.hiresScale,
      hr_steps: imageParams.hiresSteps,
      hr_upscaler: imageParams.hiresUpscaler,
      negative_prompt: imageParams.negPrompt,
      override_settings: { CLIP_stop_at_last_layers: imageParams.clipSkip ?? 1 },
      override_settings_restore_afterwards: true,
      prompt: imageParams.prompt,
      restore_faces: imageParams.restoreFaces,
      sampler_name: imageParams.sampler,
      save_images: false,
      seed: imageParams.seed,
      send_images: true,
      steps: imageParams.steps,
      subseed: imageParams.subseed ?? -1,
      subseed_strength: imageParams.subseedStrength ?? 0,
      width: imageParams.width,
    };
  }

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

  startProgress() {
    this.progress.start(1, 0, { formattedETA: "N/A" });

    this.progressInterval = setInterval(async () => {
      const res = await API.Auto1111.get("progress");
      if (!res.success) return;
      this.progress.update(Math.min(0.99, res.data.progress), {
        formattedETA: `${round(+res.data.eta_relative)}s`,
      });
    }, 500);
  }

  stopProgress() {
    clearInterval(this.progressInterval);
    this.progressInterval = null;
    this.progress.stop();
  }

  async reproduce(txt2ImgRequest: Txt2ImgRequest, imageParams: ImageParams) {
    try {
      const { imageFileName, paramFileName } = extendFileName(imageParams.fileName);

      const res = await this.txt2Img(txt2ImgRequest, imageParams);

      const { percentDiff, pixelDiff } = await compareImage(imageFileName, res.imageBuffer);
      const isReproducible = percentDiff < REPROD_DIFF_TOLERANCE;
      const chalkColor = isReproducible ? "green" : "yellow";
      console.log(
        `\nPixel Diff: ${chalk[chalkColor](pixelDiff)}. Percent diff: ${
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
        fs.writeFile(path.join(productsDir, path.basename(imageFileName)), res.imageBuffer),
        fs.writeFile(path.join(productsDir, path.basename(paramFileName)), res.params),
      ]);

      await Promise.all(
        [imageFileName, paramFileName].map((fileName) =>
          fs.rename(fileName, path.join(parentDir, path.basename(fileName)))
        )
      );

      console.log(
        `\n${chalk.cyan(imageParams.fileName)} moved to ${chalk[
          isReproducible ? "green" : "yellow"
        ](parentDir)}.`
      );
    } catch (err) {
      this.stopProgress();
      console.error(chalk.red(err.stack));
    }
  }

  async restoreFaces(extrasRequest: ExtrasRequest, withThrow = false) {
    const res = await API.Auto1111.post("extra-single-image", {
      headers: { "Content-Type": "application/json" },
      data: extrasRequest,
    });

    if (!res.success) {
      const errorMsg = `Failed to restore faces: ${res.error}`;
      if (withThrow) throw new Error(errorMsg);
      else {
        console.warn(chalk.yellow(errorMsg));
        return { success: false, error: res.error };
      }
    }

    const imageBase64: string = res.data?.image;
    return { success: true, imageBase64 };
  }

  async switchModelIfNeeded(modelName: string, models: Model[], fileName: string) {
    if (modelName !== this.getActiveModel()) {
      if (!models.map((m) => m.name).includes(modelName))
        throw new Error(`Model ${chalk.magenta(modelName)} does not exist. Skipping ${fileName}.`);

      console.log(`Setting active model to ${chalk.magenta(modelName)}...`);
      await setActiveModel(modelName);
      this.setActiveModel(modelName);

      console.log(chalk.green("Active model updated."));
    }
  }

  async switchVAEIfNeeded(vaeHash: string, vaes: VAE[]) {
    const activeVAEHash = this.getActiveVAEHash();
    if (vaeHash !== activeVAEHash) {
      if (
        ["Automatic", "None"].includes(vaeHash) &&
        !["Automatic", "None"].includes(activeVAEHash)
      ) {
        console.log(`Setting active VAE to ${chalk.magenta(vaeHash)}...`);
        await setActiveVAE(vaeHash);
        this.setActiveVAEHash(vaeHash);
      } else {
        const vae = vaes.find((v) => v.hash === vaeHash);
        if (!vae) {
          console.warn(
            chalk.yellow(
              `VAE with hash ${chalk.magenta(
                vaeHash
              )} does not exist. Setting active VAE to ${chalk.magenta("None")}...`
            )
          );

          await setActiveVAE("None");
          this.setActiveVAEHash("None");
        } else {
          console.log(`Setting active VAE to ${chalk.magenta(`${vae.fileName} (${vae.hash})`)}...`);
          await setActiveVAE(vae.fileName);
          this.setActiveVAEHash(vae.hash);
        }
      }

      console.log(chalk.green("Active VAE updated."));
    }
  }

  private async txt2Img(txt2ImgRequest: Txt2ImgRequest, imageParams: ImageParams) {
    this.startProgress();
    const txt2ImgRes = await API.Auto1111.post("txt2img", {
      headers: { "Content-Type": "application/json" },
      data: { ...txt2ImgRequest, restore_faces: false },
    });
    this.stopProgress();

    if (!txt2ImgRes.success) throw new Error(txt2ImgRes.error);
    let imageBase64 = txt2ImgRes.data?.images[0];

    if (imageParams.restoreFaces) {
      const res = await this.restoreFaces({
        codeformer_visibility: imageParams.restoreFacesStrength,
        image: imageBase64,
      });
      if (res.success) imageBase64 = res.imageBase64;
    }

    const { negTemplate, template } = imageParams;
    return {
      imageBase64: imageBase64,
      imageBuffer: Buffer.from(imageBase64, "base64"),
      params:
        (JSON.parse(txt2ImgRes.data?.info).infotexts[0] as string) +
        (template ? `\nTemplate: ${template}` : "") +
        (negTemplate ? `\nNegative Template: ${negTemplate}` : ""),
    };
  }

  async upscale(txt2ImgRequest: Txt2ImgRequest, imageParams: ImageParams) {
    try {
      const outputDir = DIR_NAMES.upscaled;
      const sourcesDir = DIR_NAMES.upscaleCompleted;
      await Promise.all([outputDir, sourcesDir].map((dir) => fs.mkdir(dir, { recursive: true })));

      const { imageFileName, paramFileName } = extendFileName(imageParams.fileName);
      const res = await this.txt2Img(txt2ImgRequest, imageParams);
      await Promise.all([
        fs.writeFile(path.join(outputDir, path.basename(imageFileName)), res.imageBuffer),
        fs.writeFile(path.join(outputDir, path.basename(paramFileName)), res.params),
      ]);

      await Promise.all(
        [imageFileName, paramFileName].map((fileName) =>
          fs.rename(fileName, path.join(sourcesDir, path.basename(fileName)))
        )
      );
    } catch (err) {
      this.stopProgress();
      console.error(chalk.red(err.stack));
    }
  }
}

const GenQueue = new GenerationQueue();

exitHook(async (callback) => {
  if (GenQueue.isPending()) await interruptGeneration(true);
  callback();
});

export const generateImages = async ({
  mode,
  imageFileNames,
  paramFileNames,
}: FileNames & { mode: Txt2ImgMode }) => {
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
    if (totalCount === 0) mainEmitter.emit("done");

    allImageParams.forEach((imageParams) =>
      GenQueue.add(async () => {
        try {
          const iterationPerfStart = performance.now();

          const { fileName, model, vaeHash } = imageParams;
          if (!imageFileNames.find((f) => f === fileName)) {
            throw new Error(`Image for ${chalk.yellow(fileName)} does not exist. Skipping.`);
          }

          const txt2ImgRequest = GenQueue.createTxt2ImgRequest(imageParams, mode);
          console.log(
            `${mode === "upscale" ? "Upscaling" : "Reproducing"} ${chalk.cyan(
              fileName
            )} with params:`,
            chalk.cyan(JSON.stringify(txt2ImgRequest, null, 2))
          );

          await GenQueue.switchModelIfNeeded(model, modelNames, fileName);
          await GenQueue.switchVAEIfNeeded(vaeHash, vaes);

          if (mode === "reproduce") await GenQueue.reproduce(txt2ImgRequest, imageParams);
          else await GenQueue.upscale(txt2ImgRequest, imageParams);

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
          GenQueue.stopProgress();
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
  const trainingParams = await loadLoraTrainingParams(DEFAULTS.LORA_PARAMS_PATH);

  const curDirName = path.basename(await fs.realpath("."));
  const loraName =
    (await prompt(chalk.blueBright(`Enter Lora name ${chalk.grey(`(${curDirName})`)}: `))) ||
    curDirName;
  trainingParams.output_name = loraName;

  const modelList = await listModels();
  const modelIndex = +(await prompt(
    `${chalk.blueBright(
      `Enter model to train on (1 - ${modelList.length}) ${chalk.grey(`(${DEFAULTS.LORA_MODEL})`)}:`
    )}\n${makeConsoleList(
      modelList.map((m) => m.name),
      true
    )}\n`
  ));
  trainingParams.pretrained_model_name_or_path =
    modelIndex > 0 && modelIndex <= modelList.length
      ? modelList[modelIndex - 1].path
      : DEFAULTS.LORA_MODEL;

  const { imageFileNames } = await listImageAndParamFileNames();

  const trainingSteps = Math.ceil(Math.max(100, 1500 / imageFileNames.length));
  const imagesDirName = `input\\${trainingSteps}_${loraName}`;
  await Promise.all(
    [imagesDirName, "logs", "output"].map((dirName) => fs.mkdir(dirName, { recursive: true }))
  );

  trainingParams.logging_dir = path.resolve(LORA.LOGS_DIR);
  trainingParams.output_dir = path.resolve(LORA.OUTPUT_DIR);
  trainingParams.train_data_dir = path.resolve(LORA.INPUT_DIR);

  await Promise.all(
    imageFileNames.map((fileName) =>
      fs.rename(`${fileName}.${EXTS.IMAGE}`, path.join(imagesDirName, `${fileName}.${EXTS.IMAGE}`))
    )
  );
  console.log(
    `Moved ${chalk.cyan(imageFileNames.length)} images to ${chalk.magenta(imagesDirName)}.`
  );

  const trainingParamsJson = JSON.stringify(trainingParams, null, 2);
  await fs.writeFile(LORA.TRAINING_PARAMS_FILE_NAME, trainingParamsJson);

  console.log(
    chalk.green(`Created ${LORA.TRAINING_PARAMS_FILE_NAME}:`),
    chalk.cyan(trainingParamsJson)
  );
  mainEmitter.emit("done");
};

/* ---------------------- Generate Txt2Img Overrides ---------------------- */
export const generateTxt2ImgOverrides = async () => {
  const overrides: Partial<Record<Txt2ImgOverride, any>> = await loadTxt2ImgOverrides();

  const booleanPrompt = async (question: string, optName: Txt2ImgOverride) => {
    const res = (await prompt(
      chalk.blueBright(`${question} (y/n)${makeExistingValueLog(overrides[optName])}`)
    )) as "y" | "n";

    if (res === "y" || res === "n") overrides[optName] = res === "y";
  };

  const numericalPrompt = async (question: string, optName: Txt2ImgOverride) => {
    const res = await prompt(
      `${chalk.blueBright(question)}${makeExistingValueLog(overrides[optName])}`
    );

    if (res.length > 0 && !isNaN(+res)) overrides[optName] = +res;
  };

  const numListPrompt = async (
    label: string,
    optName: Txt2ImgOverride,
    options: { label: string; value: string }[]
  ) => {
    const index = +(await prompt(
      `${chalk.blueBright(
        `Enter ${label} (1 - ${options.length})${makeExistingValueLog(overrides[optName])}`
      )}\n${makeConsoleList(
        options.map((o) => o.label),
        true
      )}\n`
    ));

    if (index > 0 && index <= options.length) overrides[optName] = options[index - 1].value;
  };

  const [models, samplers, upscalers, vaes] = await Promise.all([
    (await listModels()).map((m) => m.name),
    listSamplers(),
    listUpscalers(),
    listVAEs(),
  ]);

  await numericalPrompt("Enter CFG Scale", "cfgScale");
  await numericalPrompt("Enter Clip Skip", "clipSkip");
  await numericalPrompt("Enter Hires Denoising Strength", "hiresDenoisingStrength");
  await numericalPrompt("Enter Hires Scale", "hiresScale");
  await numericalPrompt("Enter Hires Steps", "hiresSteps");
  await numListPrompt("Model", "model", valsToOpts(models));
  await booleanPrompt("Restore Faces?", "restoreFaces");
  if (overrides.restoreFaces)
    await numericalPrompt("Restore Faces Strength", "restoreFacesStrength");
  await numListPrompt("Sampler", "sampler", valsToOpts(samplers));
  await numericalPrompt("Enter Seed", "seed");
  await numericalPrompt("Enter Steps", "steps");
  await numericalPrompt("Enter Subseed", "subseed");
  await numericalPrompt("Enter Subseed Strength", "subseedStrength");
  await numListPrompt("Upscaler", "hiresUpscaler", valsToOpts(upscalers));
  await numListPrompt(
    "VAE",
    "vae",
    valsToOpts(["None", "Automatic"]).concat(
      vaes.map((v) => ({ label: `${v.fileName} [${v.hash}]`, value: v.hash }))
    )
  );

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

    const config: Auto1111FolderConfig = await loadAuto1111FolderConfig();

    const promptForSymlink = async (
      folderName: Auto1111FolderName,
      targetDirName: string,
      sourceDir: string
    ) => {
      const existingPath = config[folderName];
      const folder = await promptForPath(
        `Enter path of external ${chalk.magenta(targetDirName)} folder${makeExistingValueLog(
          existingPath
        )}`
      );

      if (!folder.length && !existingPath)
        console.warn(chalk.yellow(`Not linking ${chalk.magenta(targetDirName)} folder.`));
      else {
        const hasSourceDir = await doesPathExist(sourceDir);
        if (hasSourceDir)
          await fs.rename(
            sourceDir,
            path.join(path.dirname(sourceDir), `__OLD__${path.basename(sourceDir)}`)
          );

        const targetPath = folder || existingPath;
        await fs.symlink(targetPath, path.join(rootDir, sourceDir), "junction");
        config[folderName] = targetPath;

        console.log(chalk.green(`${targetDirName} folder ${existingPath ? "re-" : ""}linked.`));
      }
    };

    await promptForSymlink("embeddings", "Embeddings", "embeddings");
    await promptForSymlink("extensions", "Extensions", "extensions");
    await promptForSymlink("lora", "Lora", "models\\Lora");
    await promptForSymlink("lycoris", "LyCORIS / LoCon / LoHa", "models\\LyCORIS");
    await promptForSymlink("models", "Models", "models\\Stable-diffusion");
    await promptForSymlink("outputs", "Outputs", "outputs");
    await promptForSymlink("vae", "VAE", "models\\VAE");
    await promptForSymlink("controlNetPoses", "ControlNet Poses", "models\\ControlNet");

    const configJson = JSON.stringify(config, null, 2);
    await fs.writeFile(AUTO1111_FOLDER_CONFIG_FILE_NAME, configJson);

    console.log(
      chalk.green(`Created ${AUTO1111_FOLDER_CONFIG_FILE_NAME}:`),
      chalk.cyan(configJson)
    );
    mainEmitter.emit("done");
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
        const fileName = `${p}.${EXTS.PARAMS}`;
        await fs.rename(fileName, path.join(dirName, path.basename(fileName)));
        unusedParams.push(p);
        console.log(`${chalk.yellow(fileName)} pruned.`);
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

export const restoreFaces = async ({ imageFileNames }: ImageFileNames) => {
  try {
    const perfStart = performance.now();

    let completedCount = 0;
    let errorCount = 0;
    const totalCount = imageFileNames.length;
    if (totalCount === 0) mainEmitter.emit("done");

    const overridesTree = await createOverridesTree(
      imageFileNames.map((f) => `${f}.${EXTS.IMAGE}`)
    );

    overridesTree.forEach(({ filePaths, overrides }) =>
      filePaths.forEach((filePath) => {
        const fileName = path.basename(filePath, `.${EXTS.IMAGE}`);

        GenQueue.add(async () => {
          const iterationPerfStart = performance.now();

          const handleCompletion = () => {
            console.log(
              `${chalk.cyan(completedCount + errorCount)} / ${chalk.grey(
                totalCount
              )} completed in ${makeTimeLog(performance.now() - iterationPerfStart)}.\n${chalk.grey(
                "-".repeat(100)
              )}`
            );

            if (completedCount + errorCount === totalCount) {
              const totalTimeElapsed = performance.now() - perfStart;
              console.log(
                `Completed: ${chalk.green(completedCount)}. Errors: ${chalk.red(
                  errorCount
                )}. Total time: ${makeTimeLog(totalTimeElapsed)}.`
              );
              mainEmitter.emit("done");
            }
          };

          try {
            console.log(`Restoring faces for ${chalk.cyan(fileName)}...`);

            const image = await fs.readFile(`${fileName}.${EXTS.IMAGE}`, { encoding: "base64" });
            const res = await GenQueue.restoreFaces({
              codeformer_visibility:
                overrides.restoreFacesStrength ?? DEFAULTS.RESTORE_FACES_STRENGTH,
              image,
            });

            if (!res.success) {
              errorCount++;
              throw new Error(res.error);
            } else {
              const name = path.basename(`${fileName}.${EXTS.IMAGE}`);
              const filePath = path.join(DIR_NAMES.restoredFaces, name);
              await fs.mkdir(path.dirname(filePath), { recursive: true });
              await fs.writeFile(filePath, Buffer.from(res.imageBase64, "base64"));

              completedCount++;
              handleCompletion();
            }
          } catch (err) {
            GenQueue.stopProgress();
            console.error(chalk.red(err.stack));
            handleCompletion();
          }
        });
      })
    );
  } catch (err) {
    console.error(chalk.red(err.stack));
  }
};

/* -------------------------- Segment by Dimensions ------------------------- */
export const segmentByDimensions = async ({ imageFileNames, noEmit }: ImageFileNames & NoEmit) => {
  console.log("Segmenting by dimensions...");

  await Promise.all(
    imageFileNames.map(async (name) => {
      try {
        const { height, width } = await sharp(`${name}.${EXTS.IMAGE}`).metadata();
        const dirName = path.join(`${height} x ${width}`, path.dirname(name));
        await fs.mkdir(dirName, { recursive: true });

        await Promise.all(
          [EXTS.IMAGE, EXTS.PARAMS].map((ext) => {
            const fileName = `${name}.${ext}`;
            return fs.rename(fileName, path.join(dirName, path.basename(fileName)));
          })
        );

        console.log(`Moved ${chalk.cyan(name)} to ${chalk.magenta(dirName)}.`);
      } catch (err) {
        console.error(chalk.red("Error segmenting file by dimensions: ", err.stack));
      }
    })
  );

  await removeEmptyFolders();

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
        chalk.blueBright(
          `Enter keywords - ${type === "all" ? "all" : "at least one"} required ${chalk(
            "(delineated by commas)"
          )}: `
        )
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
      try {
        const testKeywords = (keywords: string[], type: "every" | "some") =>
          keywords.length > 0
            ? keywords[type]((keyword) =>
                new RegExp(escapeRegEx(keyword), "im").test(imageParams.rawParams)
              )
            : true;

        const hasAllRequired = testKeywords(requiredAllKeywords, "every");
        const hasAnyRequired = testKeywords(requiredAnyKeywords, "some");

        if (hasAllRequired && hasAnyRequired) {
          const reqAll =
            requiredAllKeywords.length > 0 ? `[${requiredAllKeywords.join(" & ")}]` : "";
          const reqAny =
            requiredAnyKeywords.length > 0 ? `(${requiredAnyKeywords.join(" ~ ")})` : "";
          const dirName = `${reqAll}${reqAll && reqAny ? " - " : ""}${reqAny}`;
          await fs.mkdir(dirName, { recursive: true });

          await Promise.all(
            [EXTS.IMAGE, EXTS.PARAMS].map((ext) => {
              const filePath = `${imageParams.fileName}.${ext}`;
              const fileName = `${path.basename(imageParams.fileName)}.${ext}`;
              return fs.rename(filePath, path.join(dirName, fileName));
            })
          );

          segmentedCount++;
          console.log(`Moved ${chalk.cyan(imageParams.fileName)} to ${chalk.magenta(dirName)}.`);
        }
      } catch (err) {
        console.error(chalk.red("Error segmenting file by keywords: ", err.stack));
      }
    })
  );

  await removeEmptyFolders();

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
    allImageParams.map(async ({ fileName, model }) => {
      const dirName = path.dirname(fileName);
      await fs.mkdir(path.join(model, dirName), { recursive: true });

      await Promise.all(
        [EXTS.IMAGE, EXTS.PARAMS].map((ext) => {
          const name = `${fileName}.${ext}`;
          return fs.rename(name, path.join(model, name));
        })
      );

      console.log(`Moved ${chalk.cyan(fileName)} to ${chalk.magenta(model)}.`);
    })
  );

  await removeEmptyFolders();

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
    paramFileNames.map(async (fileName) => {
      const imageParams = await fs.readFile(`${fileName}.${EXTS.PARAMS}`);
      const targetPath = imageParams.includes("Hires upscaler")
        ? DIR_NAMES.upscaled
        : DIR_NAMES.nonUpscaled;
      const isUpscaled = targetPath === DIR_NAMES.upscaled;
      isUpscaled ? upscaledCount++ : nonUpscaledCount++;

      const filePath = path.join(targetPath, fileName);
      await fs.mkdir(path.dirname(filePath), { recursive: true });

      await Promise.all(
        [EXTS.IMAGE, EXTS.PARAMS].map((ext) =>
          fs.rename(`${fileName}.${ext}`, `${filePath}.${ext}`)
        )
      );
      console.log(
        `Moved ${chalk.cyan(fileName)} to ${(isUpscaled ? chalk.green : chalk.yellow)(targetPath)}.`
      );
    })
  );

  await removeEmptyFolders();

  console.log(
    `${chalk.green("Files segmented by upscaled.")} ${chalk.cyan(
      upscaledCount
    )} upscaled images. ${chalk.yellow(nonUpscaledCount)} non-upscaled images.`
  );
  if (!noEmit) mainEmitter.emit("done");
};

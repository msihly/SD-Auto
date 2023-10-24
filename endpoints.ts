import fs from "fs-extra";
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
  FILE_AND_DIR_NAMES,
  FILE_NAMES,
  LORA,
  OUTPUT_DIR,
  REPROD_DIFF_TOLERANCE,
  RESTORE_FACES_TYPES,
  RestoreFacesType,
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
  Img2ImgRequest,
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
  PromiseQueue,
  TreeNode,
  chunkArray,
  compareImage,
  convertImagesToJPG,
  createTree,
  delimit,
  dirToFilePaths,
  doesPathExist,
  dotKeysToTree,
  escapeRegEx,
  extendFileName,
  getImagesInFolders,
  listImagesFoundInOtherFolders,
  makeConsoleList,
  makeDirs,
  makeExistingValueLog,
  makeFilePathInDir,
  makeTimeLog,
  moveFile,
  moveFilesToFolder,
  prompt,
  randomSort,
  removeEmptyFolders,
  round,
  sha256File,
  sleep,
  valsToOpts,
  writeFilesToFolder,
} from "./utils";
import env from "./env";

/* -------------------------------------------------------------------------- */
/*                                    UTILS                                   */
/* -------------------------------------------------------------------------- */
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
    console.error(
      chalk.red(`[API Error::${method}] ${endpoint} - ${JSON.stringify(config, null, 2)}\n`, errMsg)
    );
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

  const modelHash = res.data?.sd_checkpoint_hash as string;
  const model = (await listModels()).find((m) => m.hash === modelHash)?.name;
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
export const listImageAndParamFileNames = async (
  withRecursiveFiles = false,
  exceptions = [DIR_NAMES.NON_UPSCALED, DIR_NAMES.REPRODUCIBLE, FILE_NAMES.ORIGINAL]
) => {
  console.log(`Reading files${withRecursiveFiles ? " (recursively)" : ""}...`);

  const regExNames = FILE_AND_DIR_NAMES.filter((name) => !exceptions.includes(name));
  const hasExclusions = !!regExNames.length;
  const regExFilter = hasExclusions
    ? new RegExp(regExNames.map((name) => escapeRegEx(name)).join("|"), "im")
    : null;

  const allFiles = await (withRecursiveFiles ? dirToFilePaths(".") : fs.readdir("."));
  const files = hasExclusions
    ? allFiles.filter((filePath) => !regExFilter.test(filePath))
    : allFiles;

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
      filePaths: cur.fileNames.map((fileName) => makeFilePathInDir(cur.dirPath, fileName)),
      overrides: cur.overrides,
    });
    return acc;
  }, [] as { filePaths: string[]; overrides: Txt2ImgOverrides }[]);
};

const loadTxt2ImgOverrides = async (dirPath: string = ".") => {
  try {
    const filePath = path.resolve(dirPath, TXT2IMG_OVERRIDES_FILE_NAME);
    if (!(await doesPathExist(filePath))) return {};
    else {
      return dotKeysToTree<Txt2ImgOverrides>(
        JSON.parse(await fs.readFile(filePath, { encoding: "utf8" }))
      );
    }
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

    const imageParams = await fs.readFile(extendFileName(fileName, EXTS.PARAMS), {
      encoding: "utf8",
    });

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
        `Invalid model name ${chalk.yellow(rawModelName)} found for ${chalk.cyan(paramFileName)}.`
      );
    }

    /* ------------------------------ Main Settings ----------------------------- */
    const cfgScale = overrides?.cfgScale ?? parseImageParam(restParams, "CFG scale", true);
    const clipSkip = overrides?.clipSkip ?? parseImageParam(restParams, "Clip skip", true, true);
    const hiresDenoisingStrength =
      overrides?.hiresDenoisingStrength ??
      parseImageParam(restParams, "Hires denoising strength", true, true);
    const hiresScale =
      overrides?.hiresScale ?? parseImageParam(restParams, "Hires scale", true, true);
    const hiresSteps =
      overrides?.hiresSteps ?? parseImageParam(restParams, "Hires steps", true, true);
    let hiresUpscaler = parseImageParam(restParams, "Hires upscaler", false, true);
    const isUpscaled = hiresUpscaler !== undefined;
    hiresUpscaler = overrides?.hiresUpscaler ?? hiresUpscaler;

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

    /* ---------------------------- "ADetailer" Extension ---------------------------- */
    /* ----------------- "Multi Diffusion / Tiled VAE" Extension ---------------- */
    const aDetailer = overrides.aDetailer
      ? {
          ...overrides.aDetailer,
          confidence:
            overrides.aDetailer?.confidence ??
            parseImageParam(restParams, "ADetailer confidence", true, true),
          denoisingStrength:
            overrides.aDetailer?.denoisingStrength ??
            parseImageParam(restParams, "ADetailer denoising strength", true, true),
          dilateErode:
            overrides.aDetailer?.dilateErode ??
            parseImageParam(restParams, "ADetailer dilate/erode", true, true),
          enabled: true,
          inpaintOnlyMasked:
            overrides.aDetailer?.inpaintOnlyMasked ??
            parseImageParam(restParams, "ADetailer inpaint only masked", false, true) !== "False",
          inpaintPadding:
            overrides.aDetailer?.inpaintPadding ??
            parseImageParam(restParams, "ADetailer inpaint padding", true, true),
          maskBlur:
            overrides.aDetailer?.maskBlur ??
            parseImageParam(restParams, "ADetailer mask blur", true, true),
          maskOnlyTopKLargest:
            overrides.aDetailer?.maskOnlyTopKLargest ??
            parseImageParam(restParams, "ADetailer mask_only_top_k_largest", true, true),
          model:
            overrides.aDetailer.model ??
            parseImageParam(restParams, "ADetailer model", false, true) ??
            DEFAULTS.ADETAILER_MODEL,
        }
      : undefined;

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
            enabled: overrides.tiledDiffusion?.enabled ?? env.TILED_DIFFUSION.enabled,
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
            enabled: overrides.tiledVAE?.enabled ?? env.TILED_VAE.enabled,
            encoderTileSize: overrides.tiledVAE?.encoderTileSize ?? env.TILED_VAE.encoderTileSize,
            fastDecoderEnabled:
              overrides.tiledVAE?.fastDecoderEnabled ?? env.TILED_VAE.fastDecoderEnabled,
            fastEncoderEnabled:
              overrides.tiledVAE?.fastEncoderEnabled ?? env.TILED_VAE.fastEncoderEnabled,
            vaeToGPU: overrides.tiledVAE?.vaeToGPU ?? env.TILED_VAE.vaeToGPU,
          }
        : undefined;

    return {
      aDetailer,
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
      isUpscaled,
      model,
      modelHash,
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
    if (err.code === "EMFILE")
      return sleep(2000).then(() => parseImageParams({ models, overrides, paramFileName }));

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

  const overridesTree = await createOverridesTree(
    paramFileNames.map((f) => extendFileName(f, EXTS.PARAMS))
  );

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
      const modelSort = a.model.localeCompare(b.model);
      if (modelSort !== 0) return modelSort;

      if (a.vaeHash && !b.vaeHash) return -1;
      if (!a.vaeHash && b.vaeHash) return 1;
      if (a.vaeHash && b.vaeHash) return a.vaeHash.localeCompare(b.vaeHash);
      return 0;
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
      `${chalk.blueBright("Enter batch size")} ${chalk.grey(`(${DEFAULTS.BATCH_SIZE})`)}: `
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
const remapModelName = (models: { hash: string; name: string }[], modelHash: string) => {
  try {
    return models.find((m) => m.hash === modelHash)?.name;
  } catch (err) {
    console.warn(chalk.yellow("Missing 'models' param in remapModelName!"));
    return undefined;
  }
};

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
  await Promise.all(nonJPG.map((fileName) => moveFile(fileName, DIR_NAMES.NON_JPG)));
  console.log(chalk.green(`Converted ${chalk.cyan(nonJPG.length)} files to JPG.`));
  mainEmitter.emit("done");
};

/* ----------------------------- Flatten Folders ---------------------------- */
export const flattenFolders = async () => {
  const depth = +(await prompt(chalk.blueBright("Depth: ")));
  if (isNaN(depth)) throw new Error("Depth is not a number!");

  const filePaths = await dirToFilePaths(".", true);

  const dirPaths = filePaths.reduce((acc, cur) => {
    const pathParts = cur.split(path.sep);
    if (pathParts.length > depth) {
      const targetDirPath = path.join(...pathParts.slice(0, depth));
      const dir = acc.find((o) => o.dirPath === targetDirPath);
      if (dir) dir.filePaths.push(cur);
      else acc.push({ dirPath: targetDirPath, filePaths: [cur] });
    }
    return acc;
  }, [] as { dirPath: string; filePaths: string[] }[]);

  await Promise.all(
    dirPaths.flatMap((d) =>
      d.filePaths.map((filePath) =>
        moveFilesToFolder({
          filePath,
          folderName: d.dirPath,
          newName: path.basename(filePath),
          withConsole: true,
        })
      )
    )
  );

  await removeEmptyFolders();
  mainEmitter.emit("done");
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

  createRequest(imageParams: ImageParams) {
    const hiresScale = imageParams.hiresScale ?? DEFAULTS.HIRES_SCALE;
    const hiresUpscaler = imageParams.hiresUpscaler ?? DEFAULTS.HIRES_UPSCALER;

    return {
      alwayson_scripts: {
        ADetailer: imageParams.aDetailer?.enabled
          ? {
              args: [
                true, // enabled
                true, // skip img2img
                {
                  ad_cfg_scale: imageParams.aDetailer.cfgScale ?? imageParams.cfgScale,
                  ad_clip_skip: imageParams.aDetailer.clipSkip ?? imageParams.clipSkip,
                  ad_confidence: imageParams.aDetailer.confidence,
                  ad_controlnet_guidance_end: imageParams.aDetailer.controlnetGuidanceEnd,
                  ad_controlnet_guidance_start: imageParams.aDetailer.controlnetGuidanceStart,
                  ad_controlnet_model: imageParams.aDetailer.controlnetModel,
                  ad_controlnet_module: imageParams.aDetailer.controlnetModule,
                  ad_controlnet_weight: imageParams.aDetailer.controlnetWeight,
                  ad_denoising_strength: imageParams.aDetailer.denoisingStrength,
                  ad_dilate_erode: imageParams.aDetailer.dilateErode,
                  ad_inpaint_height: imageParams.aDetailer.inpaintHeight,
                  ad_inpaint_only_masked: imageParams.aDetailer.inpaintOnlyMasked,
                  ad_inpaint_only_masked_padding: imageParams.aDetailer.inpaintPadding,
                  ad_inpaint_width: imageParams.aDetailer.inpaintWidth,
                  ad_mask_blur: imageParams.aDetailer.maskBlur,
                  ad_mask_k_largest: imageParams.aDetailer.maskOnlyTopKLargest,
                  ad_mask_max_ratio: imageParams.aDetailer.maskMaxRatio,
                  ad_mask_merge_invert: imageParams.aDetailer.maskMergeInvert,
                  ad_mask_min_ratio: imageParams.aDetailer.maskMinRatio,
                  ad_model: imageParams.aDetailer.model ?? DEFAULTS.ADETAILER_MODEL,
                  ad_negative_prompt: imageParams.aDetailer.negPrompt,
                  ad_noise_multiplier: imageParams.aDetailer.noiseMultiplier,
                  ad_prompt: imageParams.aDetailer.prompt,
                  ad_restore_face: imageParams.aDetailer.restoreFace,
                  ad_sampler: imageParams.aDetailer.sampler ?? imageParams.sampler,
                  ad_steps:
                    imageParams.aDetailer.steps ?? imageParams.hiresSteps ?? imageParams.steps,
                  ad_use_cfg_scale: imageParams.aDetailer.useCfgScale,
                  ad_use_clip_skip: imageParams.aDetailer.useClipSkip,
                  ad_use_inpaint_width_height: imageParams.aDetailer.useInpaintWidthHeight,
                  ad_use_noise_multiplier: imageParams.aDetailer.useNoiseMultiplier,
                  ad_use_sampler: imageParams.aDetailer.useSampler,
                  ad_use_steps: imageParams.aDetailer.useSteps,
                  ad_x_offset: imageParams.aDetailer.xOffset,
                  ad_y_offset: imageParams.aDetailer.yOffset,
                },
              ],
            }
          : undefined,
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
        "Tiled Diffusion": imageParams.tiledDiffusion.enabled
          ? {
              args: [
                "True", // enabled
                imageParams.tiledDiffusion.method, // "MultiDiffusion"
                imageParams.tiledDiffusion.overwriteSize, // "False"
                imageParams.tiledDiffusion.keepInputSize, // "True"
                imageParams.width,
                imageParams.height,
                imageParams.tiledDiffusion.tileWidth, // 128
                imageParams.tiledDiffusion.tileHeight, // 128
                imageParams.tiledDiffusion.tileOverlap, // 48
                imageParams.tiledDiffusion.batchSize, // 4
                hiresUpscaler,
                hiresScale,
              ],
            }
          : undefined,
        "Tiled VAE": imageParams.tiledVAE.enabled
          ? {
              args: [
                "True", // enabled
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
      denoising_strength: imageParams.hiresDenoisingStrength ?? DEFAULTS.HIRES_DENOISING_STRENGTH,
      enable_hr: false,
      height: imageParams.height,
      hr_scale: hiresScale,
      hr_steps: imageParams.hiresSteps,
      hr_upscaler: hiresUpscaler,
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

  async createImg2ImgRequest(
    imageBase64: string,
    imageParams: ImageParams
  ): Promise<Img2ImgRequest> {
    const request = this.createRequest(imageParams);
    return { ...request, init_images: [imageBase64], tiling: false };
  }

  createTxt2ImgRequest(imageParams: ImageParams, mode: Txt2ImgMode): Txt2ImgRequest {
    const request = this.createRequest(imageParams);
    return { ...request, enable_hr: mode === "upscale" };
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
      const res = await this.txt2Img(txt2ImgRequest, imageParams);

      const filePath = imageParams.fileName;
      const imageFileName = extendFileName(filePath, EXTS.IMAGE);

      const { percentDiff, pixelDiff } = await compareImage(imageFileName, res.imageBuffer);
      const isReproducible = percentDiff < REPROD_DIFF_TOLERANCE;
      const chalkColor = isReproducible ? "green" : "yellow";
      console.log(
        `\nPixel Diff: ${chalk[chalkColor](pixelDiff)}. Percent diff: ${
          chalk[chalkColor](round(percentDiff * 100)) + "%"
        }.`
      );

      const prefix = `${imageParams.modelHash}-${imageParams.seed}`;
      const folderName = path.join(
        isReproducible ? DIR_NAMES.REPRODUCIBLE : DIR_NAMES.NON_REPRODUCIBLE,
        prefix
      );
      const newName = `${prefix} - ${FILE_NAMES.REPRODUCED}`;

      await writeFilesToFolder({
        files: [
          { content: res.imageBuffer, ext: EXTS.IMAGE, name: newName },
          { content: res.params, ext: EXTS.PARAMS, name: newName },
        ],
        folderName,
      });

      await moveFilesToFolder({
        exts: [EXTS.IMAGE, EXTS.PARAMS],
        filePath,
        folderColor: chalkColor,
        folderName,
        newName: `${prefix} - ${FILE_NAMES.ORIGINAL}`,
        withConsole: true,
      });
    } catch (err) {
      this.stopProgress();
      console.error(chalk.red(err.stack));
    }
  }

  async restoreFaces({
    imageBase64,
    imageParams,
    strength,
    type,
    withThrow = false,
  }: {
    imageBase64: string;
    imageParams: ImageParams;
    strength?: number;
    type: RestoreFacesType;
    withThrow?: boolean;
  }) {
    const isADetailer = type === RESTORE_FACES_TYPES.ADETAILER;
    const data = isADetailer
      ? await this.createImg2ImgRequest(imageBase64, {
          ...imageParams,
          aDetailer: { denoisingStrength: strength, enabled: true },
        })
      : ({
          [type === RESTORE_FACES_TYPES.CODEFORMER ? "codeformer_visibility" : "gfpgan_visibility"]:
            strength ?? DEFAULTS.RESTORE_FACES_STRENGTH,
          image: imageBase64,
        } as ExtrasRequest);

    console.log(
      `Restoring faces for ${chalk.cyan(imageParams.fileName)} using method ${type} with params:`,
      chalk.cyan(
        JSON.stringify(
          { ...data, image: "<base64 encoded image>", init_images: ["<base64 encoded image>"] },
          null,
          2
        )
      )
    );

    this.startProgress();
    const res = await API.Auto1111.post(isADetailer ? "img2img" : "extra-single-image", {
      headers: { "Content-Type": "application/json" },
      data,
    });
    this.stopProgress();

    if (!res.success) {
      const errorMsg = `Failed to restore faces: ${res.error}`;
      if (withThrow) throw new Error(errorMsg);
      else {
        console.warn(chalk.yellow(errorMsg));
        return { success: false, error: res.error };
      }
    }

    const resImageBase64: string = res.data?.images?.[0] ?? res.data?.image;
    return { success: true, imageBase64: resImageBase64 };
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
      if (["Automatic", "None"].includes(vaeHash)) {
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
        imageBase64,
        imageParams,
        strength: imageParams.restoreFacesStrength,
        type: RESTORE_FACES_TYPES.GFPGAN,
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
      const res = await this.txt2Img(txt2ImgRequest, imageParams);

      const prefix = `${imageParams.modelHash}-${imageParams.seed}`;
      const folderName = path.join(DIR_NAMES.UPSCALED, prefix);
      const newName = `${prefix} - ${FILE_NAMES.UPSCALED}`;

      await writeFilesToFolder({
        files: [
          { content: res.imageBuffer, ext: EXTS.IMAGE, name: newName },
          { content: res.params, ext: EXTS.PARAMS, name: newName },
        ],
        folderName,
      });

      await moveFilesToFolder({
        exts: [EXTS.IMAGE, EXTS.PARAMS],
        filePath: imageParams.fileName,
        folderName,
        newName: `${prefix} - ${FILE_NAMES.ORIGINAL}`,
      });
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
            await removeEmptyFolders();

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
  await makeDirs([imagesDirName, "logs", "output"]);

  trainingParams.logging_dir = path.resolve(LORA.LOGS_DIR);
  trainingParams.output_dir = path.resolve(LORA.OUTPUT_DIR);
  trainingParams.train_data_dir = path.resolve(LORA.INPUT_DIR);

  await Promise.all(
    imageFileNames.map((name) => moveFile(extendFileName(name, EXTS.IMAGE), imagesDirName))
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

  await booleanPrompt("ADetailer?", "aDetailer.enabled");
  if (overrides["aDetailer.enabled"]) {
    await numListPrompt("- Model", "aDetailer.model", valsToOpts(["face_yolov8n.pt"]));
    await numericalPrompt("- Denoising Strength", "aDetailer.denoisingStrength");
    await numericalPrompt("- Steps", "aDetailer.steps");
    await numericalPrompt("- Mask K Largest", "aDetailer.maskOnlyTopKLargest");
  }

  await booleanPrompt("Tiled Diffusion?", "tiledDiffusion.enabled");
  if (overrides["tiledDiffusion.enabled"]) {
    await numListPrompt(
      "- Method",
      "tiledDiffusion.method",
      valsToOpts(["Mixture of Diffusers", "MultiDiffusion"])
    );

    await numericalPrompt("- Batch Size", "tiledDiffusion.batchSize");
    await numericalPrompt("- Tile Height", "tiledDiffusion.tileHeight");
    await numericalPrompt("- Tile Width", "tiledDiffusion.tileWidth");
    await numericalPrompt("- Tile Overlap", "tiledDiffusion.tileOverlap");
    await booleanPrompt("- Keep Input Size?", "tiledDiffusion.keepInputSize");
    await booleanPrompt("- Overwrite Size?", "tiledDiffusion.overwriteSize");
  }

  await booleanPrompt("Tiled VAE?", "tiledVAE.enabled");
  if (overrides["tiledVAE.enabled"]) {
    await numericalPrompt("- Encoder Tile Size", "tiledVAE.encoderTileSize");
    await numericalPrompt("- Decoder Tile Size", "tiledVAE.decoderTileSize");
    await booleanPrompt("- Color Fix Enabled?", "tiledVAE.colorFixEnabled");
    await booleanPrompt("- Fast Encoder Enabled?", "tiledVAE.fastEncoderEnabled");
    await booleanPrompt("- Fast Decoder Enabled?", "tiledVAE.fastDecoderEnabled");
    await booleanPrompt("- VAE To GPU?", "tiledVAE.vaeToGPU");
  }

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
        if (hasSourceDir) await moveFile(sourceDir, sourceDir, `${path.basename(sourceDir)} [OLD]`);

        const targetPath = folder || existingPath;
        await fs.symlink(targetPath, path.join(rootDir, sourceDir), "dir");
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

/* ------------------- Prune Images ------------------- */
export const pruneImages = async ({
  imageFileNames,
  paramFileNames,
  noEmit,
}: FileNames & NoEmit) => {
  const unusedImages: string[] = [];

  await Promise.all(
    imageFileNames.map(async (img) => {
      if (paramFileNames.find((p) => p === img)) return;

      await moveFilesToFolder({
        filePath: img,
        folderColor: "yellow",
        folderName: DIR_NAMES.PRUNED_IMAGES,
        exts: [EXTS.IMAGE],
        withConsole: true,
      });

      unusedImages.push(img);
    })
  );

  console.log(`${chalk.yellow(unusedImages.length)} unused images pruned.`);
  if (!noEmit) mainEmitter.emit("done");

  return unusedImages;
};

/* ------------------- Prune Generation Parameters ------------------- */
export const pruneImageParams = async ({
  noEmit,
  imageFileNames,
  paramFileNames,
}: FileNames & NoEmit) => {
  const unusedParams: string[] = [];

  await Promise.all(
    paramFileNames.map(async (p) => {
      if (imageFileNames.find((img) => img === p)) return;

      await moveFilesToFolder({
        filePath: p,
        folderColor: "yellow",
        folderName: DIR_NAMES.PRUNED_PARAMS,
        exts: [EXTS.PARAMS],
        withConsole: true,
      });

      unusedParams.push(p);
    })
  );

  console.log(`${chalk.yellow(unusedParams.length)} unused params pruned.`);
  if (!noEmit) mainEmitter.emit("done");

  return unusedParams;
};

/* -------------------------- Remove Empty Folders -------------------------- */
export const removeEmptyFoldersAction = async () => {
  await removeEmptyFolders();
  mainEmitter.emit("done");
};

/* ------------------------------ Restore Faces ----------------------------- */
export const restoreFaces = async ({ imageFileNames }: ImageFileNames) => {
  try {
    const perfStart = performance.now();

    let completedCount = 0;
    let errorCount = 0;

    const fileNames = imageFileNames.reduce((acc, cur) => {
      const [modelHash, seed] = path.basename(cur, path.extname(cur)).split("-");
      const existing = acc.find((f) => {
        const [fModelHash, fSeed] = path.basename(f, path.extname(f)).split("-");
        return fModelHash === modelHash && fSeed === seed;
      });

      if (!existing) acc.push(cur);
      else if (cur.includes(FILE_NAMES.UPSCALED)) acc[acc.indexOf(existing)] = cur;
      return acc;
    }, [] as string[]);

    const totalCount = fileNames.length;
    if (totalCount === 0) return mainEmitter.emit("done");

    const types = Object.values(RESTORE_FACES_TYPES) as RestoreFacesType[];
    const typeIndex = +(await prompt(
      `${chalk.blueBright("Select method: ")}\n${makeConsoleList(types, true)}\n`
    ));
    const type = types[typeIndex - 1];

    const overridesTree = await createOverridesTree(
      fileNames.map((f) => extendFileName(f, EXTS.IMAGE))
    );

    const models = await listModels();

    overridesTree.forEach(({ filePaths, overrides }) =>
      filePaths.forEach((filePath) => {
        const fileName = path.join(
          path.dirname(filePath),
          path.basename(filePath, path.extname(filePath))
        );
        const imageFileName = extendFileName(fileName, EXTS.IMAGE);
        const paramFileName = extendFileName(fileName, EXTS.PARAMS);

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
            const imageParams = await parseImageParams({ models, overrides, paramFileName });

            const res = await GenQueue.restoreFaces({
              imageBase64: await fs.readFile(imageFileName, { encoding: "base64" }),
              imageParams,
              strength: overrides.restoreFacesStrength,
              type,
            });

            if (!res.success) {
              errorCount++;
              throw new Error(res.error);
            } else {
              const prefix = `${imageParams.modelHash}-${imageParams.seed}`;
              const name = `${prefix} - ${FILE_NAMES.RESTORED_FACES} (${type})`;

              const paramsContent = await fs.readFile(extendFileName(imageFileName, EXTS.PARAMS));
              await writeFilesToFolder({
                files: [
                  { content: Buffer.from(res.imageBase64, "base64"), ext: EXTS.IMAGE, name },
                  { content: paramsContent, ext: EXTS.PARAMS, name },
                ],
                folderName: DIR_NAMES.RESTORED_FACES,
              });

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
    mainEmitter.emit("done");
  }
};

/* ----------------------------- Segment by Batches ----------------------------- */
export const segmentByBatches = async ({
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
    folderName: `[Batch - ${i + 1}] ${folderName ? `${folderName} ` : ""}(${batch.length})`,
  }));

  await Promise.all(
    batches.map(async (batch) => {
      await makeDirs([batch.folderName]);

      await Promise.all(
        batch.fileNames.map(async (name) => {
          await moveFilesToFolder({
            exts: [EXTS.IMAGE, EXTS.PARAMS],
            filePath: name,
            folderName: batch.folderName,
            withConsole: true,
          });
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

  await removeEmptyFolders();
  if (!noEmit) mainEmitter.emit("done");
};

/* -------------------------- Segment by Dimensions ------------------------- */
export const segmentByDimensions = async ({
  imageFileNames,
  paramFileNames,
  noEmit,
}: FileNames & NoEmit) => {
  const res = (await prompt(
    chalk.blueBright(
      `Use file dimensions instead of original size? (y/n) ${makeExistingValueLog("n")}`
    )
  )) as "y" | "n";
  const shouldUseFileDimensions = res === "y";

  console.log("Segmenting by dimensions...");

  const moveToDimFolder = async ({ height, name, width }) =>
    await moveFilesToFolder({
      filePath: name,
      folderName: `${height} x ${width}`,
      exts: [EXTS.IMAGE, EXTS.PARAMS],
      withConsole: true,
    });

  if (shouldUseFileDimensions) {
    await Promise.all(
      imageFileNames.map(async (name) => {
        try {
          const fileName = extendFileName(name, EXTS.IMAGE);
          const { height, width } = await sharp(fileName).metadata();
          await moveToDimFolder({ height, name, width });
        } catch (err) {
          console.error(chalk.red("Error segmenting file by dimensions: ", err.stack));
        }
      })
    );
  } else {
    const allImageParams = await parseAndSortImageParams({ imageFileNames, paramFileNames });

    await Promise.all(
      allImageParams.map(async (params) => {
        try {
          const { height, width } = params;
          await moveToDimFolder({ height, name: params.fileName, width });
        } catch (err) {
          console.error(chalk.red("Error segmenting file by dimensions: ", err.stack));
        }
      })
    );
  }

  await removeEmptyFolders();

  console.log(chalk.cyan(imageFileNames.length), chalk.green(" files segmented by dimensions."));
  if (!noEmit) mainEmitter.emit("done");
};

/* --------------------------- Segment by Filename -------------------------- */
export const segmentByFilename = async ({
  imageFileNames,
  paramFileNames,
  noEmit,
}: FileNames & NoEmit) => {
  console.log("Segmenting by filename...");

  const allImageParams = await parseAndSortImageParams({ imageFileNames, paramFileNames });

  await Promise.all(
    allImageParams.map(async (params) => {
      try {
        const filePath = params.fileName;
        const fileName = path.basename(filePath, path.extname(filePath));

        await moveFilesToFolder({
          exts: [EXTS.IMAGE, EXTS.PARAMS],
          filePath,
          folderName: fileName,
          newName: Object.values(FILE_NAMES).includes(fileName)
            ? undefined
            : `${fileName} - ${params.isUpscaled ? FILE_NAMES.UPSCALED : FILE_NAMES.ORIGINAL}`,
          withConsole: true,
        });
      } catch (err) {
        console.error(chalk.red("Error segmenting file by filename: ", err.stack));
      }
    })
  );

  await removeEmptyFolders();

  console.log(chalk.cyan(imageFileNames.length), chalk.green(" files segmented by filename."));
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

          await moveFilesToFolder({
            filePath: imageParams.fileName,
            folderName: `${reqAll}${reqAll && reqAny ? " - " : ""}${reqAny}`,
            exts: [EXTS.IMAGE, EXTS.PARAMS],
            withConsole: true,
          });

          segmentedCount++;
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
      await moveFilesToFolder({
        filePath: fileName,
        folderName: model,
        exts: [EXTS.IMAGE, EXTS.PARAMS],
        withConsole: true,
      });
    })
  );

  await removeEmptyFolders();

  console.log(chalk.cyan(allImageParams.length), chalk.green(" files segmented by model."));
  if (!noEmit) mainEmitter.emit("done");
};

/* --------------------------- Segment by ModelHash-Seed -------------------------- */
export const segmentByModelHashSeed = async ({
  imageFileNames,
  paramFileNames,
  noEmit,
}: FileNames & NoEmit) => {
  console.log("Renaming to [ModelHash]-[Seed]...");

  const allImageParams = await parseAndSortImageParams({ imageFileNames, paramFileNames });
  const suffixRegEx = new RegExp(
    `((?:${Object.values(FILE_NAMES).join("|")})((?:\\s\\((?:${Object.values(
      RESTORE_FACES_TYPES
    ).join("|")})\\))?(?:\\s\\(\\d+\\))?))(?:\\.\\w+)?$`
  );

  await Promise.all(
    allImageParams.map(async (params) => {
      try {
        const filePath = params.fileName;
        const fileName = path.basename(filePath, path.extname(filePath));
        const prefix = `${params.modelHash}-${params.seed}`;
        const suffix =
          fileName.match(suffixRegEx)?.[1] ??
          (params.isUpscaled ? FILE_NAMES.UPSCALED : FILE_NAMES.ORIGINAL);
        const newName = `${prefix} - ${suffix}`;

        await moveFilesToFolder({
          exts: [EXTS.IMAGE, EXTS.PARAMS],
          filePath,
          folderName: prefix,
          newName,
          withConsole: true,
        });
      } catch (err) {
        console.error(chalk.red("Error renaming files: ", err.stack));
      }
    })
  );

  await removeEmptyFolders();

  console.log(
    chalk.cyan(imageFileNames.length + paramFileNames.length),
    chalk.green(" files renamed.")
  );
  if (!noEmit) mainEmitter.emit("done");
};

/* --------------------------- Segment by Upscaled -------------------------- */
export const segmentByUpscaled = async ({ paramFileNames, noEmit }: ParamFileNames & NoEmit) => {
  console.log("Segmenting by upscaled...");

  let upscaledCount = 0;
  let nonUpscaledCount = 0;

  await Promise.all(
    paramFileNames.map(async (fileName) => {
      const imageParams = await fs.readFile(`${fileName}.${EXTS.PARAMS}`);
      const isUpscaled = imageParams.includes("Hires upscaler");
      const folderName = isUpscaled ? DIR_NAMES.UPSCALED : DIR_NAMES.NON_UPSCALED;
      isUpscaled ? upscaledCount++ : nonUpscaledCount++;

      await moveFilesToFolder({
        folderColor: isUpscaled ? "green" : "yellow",
        folderName,
        filePath: fileName,
        exts: [EXTS.IMAGE, EXTS.PARAMS],
        withConsole: true,
      });
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

import chalk from "chalk";
import { prompt, readline } from "./utils";
import { DIR_NAMES, FILE_AND_DIR_NAMES, FILE_NAMES } from "./constants";
import { FileNames } from "./types";
import {
  convertImagesInCurDirToJPG,
  flattenFolders,
  generateImages,
  generateLoraTrainingFolderAndParams,
  generateTxt2ImgOverrides,
  getActiveModel,
  getActiveVAE,
  initAutomatic1111Folders,
  listImageAndParamFileNames,
  listModels,
  listSamplers,
  listUpscalers,
  listVAEs,
  pruneImageParams,
  pruneImages,
  removeEmptyFoldersAction,
  restoreFaces,
  segmentByBatches,
  segmentByDimensions,
  segmentByFilename,
  segmentByKeywords,
  segmentByModel,
  segmentByModelHashSeed,
  segmentByUpscaled,
} from "./endpoints";

const SCRIPT_OPTS: {
  action: ({ imageFileNames, paramFileNames }?: FileNames) => Promise<any>;
  hasRecursiveOption?: boolean;
  nameExceptions?: string[];
  needsFiles: boolean;
  label: string;
}[] = [
  {
    action: (fileNames: FileNames) => generateImages({ ...fileNames, mode: "upscale" }),
    hasRecursiveOption: true,
    label: "Upscale (Hires Fix)",
    needsFiles: true,
  },
  {
    action: (fileNames: FileNames) => generateImages({ ...fileNames, mode: "reproduce" }),
    hasRecursiveOption: true,
    label: "Reproduce",
    needsFiles: true,
  },
  {
    action: restoreFaces,
    hasRecursiveOption: true,
    label: "Restore Faces",
    nameExceptions: [
      DIR_NAMES.NON_UPSCALED,
      DIR_NAMES.UPSCALED,
      FILE_NAMES.ORIGINAL,
      FILE_NAMES.UPSCALED,
    ],
    needsFiles: true,
  },
  {
    action: pruneImageParams,
    hasRecursiveOption: true,
    label: "Prune Generation Parameters",
    needsFiles: true,
  },
  {
    action: pruneImages,
    hasRecursiveOption: true,
    label: "Prune Images",
    needsFiles: true,
  },
  {
    action: segmentByBatches,
    hasRecursiveOption: true,
    label: "Segment by Batches",
    nameExceptions: FILE_AND_DIR_NAMES,
    needsFiles: true,
  },
  {
    action: segmentByDimensions,
    hasRecursiveOption: true,
    label: "Segment by Dimensions",
    nameExceptions: FILE_AND_DIR_NAMES,
    needsFiles: true,
  },
  {
    action: segmentByFilename,
    hasRecursiveOption: true,
    label: "Segment by Filename",
    nameExceptions: [...Object.values(DIR_NAMES)],
    needsFiles: true,
  },
  {
    action: segmentByKeywords,
    hasRecursiveOption: true,
    label: "Segment by Keywords",
    nameExceptions: FILE_AND_DIR_NAMES,
    needsFiles: true,
  },
  {
    action: segmentByModel,
    hasRecursiveOption: true,
    label: "Segment by Model",
    nameExceptions: FILE_AND_DIR_NAMES,
    needsFiles: true,
  },
  {
    action: segmentByModelHashSeed,
    hasRecursiveOption: true,
    label: "Segment by ModelHash-Seed",
    nameExceptions: FILE_AND_DIR_NAMES,
    needsFiles: true,
  },
  {
    action: segmentByUpscaled,
    hasRecursiveOption: true,
    label: "Segment by Upscaled",
    nameExceptions: FILE_AND_DIR_NAMES,
    needsFiles: true,
  },
  {
    action: generateTxt2ImgOverrides,
    label: "Generate Txt2Img Overrides",
    needsFiles: false,
  },
  {
    action: generateLoraTrainingFolderAndParams,
    label: "Generate Lora Training Folder and Params",
    needsFiles: false,
  },
  {
    action: convertImagesInCurDirToJPG,
    label: "Convert Images to JPG",
    needsFiles: false,
  },
  {
    action: () => getActiveModel(true, true),
    label: "Get Active Model",
    needsFiles: false,
  },
  {
    action: () => getActiveVAE(true, true),
    label: "Get Active VAE",
    needsFiles: false,
  },
  {
    action: () => listModels(true, true),
    label: "List Models",
    needsFiles: false,
  },
  {
    action: () => listSamplers(true, true),
    label: "List Samplers",
    needsFiles: false,
  },
  {
    action: () => listUpscalers(true, true),
    label: "List Upscalers",
    needsFiles: false,
  },
  {
    action: () => listVAEs(true, true),
    label: "List VAEs",
    needsFiles: false,
  },
  {
    action: initAutomatic1111Folders,
    label: "Initialize Automatic1111 Folders (Symlinks)",
    needsFiles: false,
  },
  {
    action: flattenFolders,
    label: "Flatten Folders",
    needsFiles: false,
  },
  {
    action: removeEmptyFoldersAction,
    label: "Remove Empty Folders",
    needsFiles: true,
  },
];

export const main = async () => {
  try {
    const input = await prompt(
      `${SCRIPT_OPTS.map(
        (opt, i) =>
          `  ${chalk.cyan(i + 1)}: ${opt.label}${
            opt.hasRecursiveOption ? chalk.magenta(" (r)") : ""
          }`
      ).join("\n")}\n${chalk.red("  0:")} Exit\n${chalk.blueBright(
        `Select an option ${chalk.grey("(--r for recursion, commas for chaining)")}: `
      )}`
    );

    const scriptIndices = input
      .replace(/[^\d,]+/gi, "")
      .split(",")
      .map((i) => +i)
      .filter((i) => !isNaN(i));
    if (!scriptIndices.length)
      throw new Error(
        `Invalid selection. Only comma-delineated numbers between 0 and ${SCRIPT_OPTS.length} allowed.`
      );

    for (const scriptIndex of scriptIndices) {
      if (isNaN(scriptIndex) || scriptIndex < 0 || scriptIndex > SCRIPT_OPTS.length)
        throw new Error(`Invalid selection. Enter a number between 0 and ${SCRIPT_OPTS.length}.`);

      if (scriptIndex === 0) return readline.close();

      const script = SCRIPT_OPTS[scriptIndex - 1];
      if (!script) throw new Error("Failed to find matching script.");

      if (script.needsFiles) {
        const withRecursion = script.hasRecursiveOption && input.includes("--r");
        const fileNames = await listImageAndParamFileNames(withRecursion, script.nameExceptions);
        await script.action(fileNames);
      } else await script.action();
    }
  } catch (err) {
    console.error(chalk.red(err));
    console.log(chalk.grey("-".repeat(100)));
    main();
  }
};

main();

import chalk from "chalk";
import { prompt, readline } from "./utils";
import { FileNames } from "./types";
import {
  convertImagesInCurDirToJPG,
  generateBatches,
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
  pruneFilesFoundInFolders,
  pruneImageParams,
  pruneParamsAndSegmentUpscaled,
  restoreFaces,
  segmentByDimensions,
  segmentByKeywords,
  segmentByModel,
  segmentByUpscaled,
} from "./endpoints";

const SCRIPT_OPTS: {
  action: ({ imageFileNames, paramFileNames }?: FileNames) => any;
  hasRecursiveOption?: boolean;
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
    needsFiles: true,
  },
  {
    action: pruneParamsAndSegmentUpscaled,
    hasRecursiveOption: true,
    label: "Prune Generation Parameters & Segment by Upscaled",
    needsFiles: true,
  },
  {
    action: pruneImageParams,
    hasRecursiveOption: true,
    label: "Prune Generation Parameters",
    needsFiles: true,
  },
  {
    action: pruneFilesFoundInFolders,
    label: "Prune Files Found in Other Folders",
    needsFiles: true,
  },
  {
    action: segmentByDimensions,
    hasRecursiveOption: true,
    label: "Segment by Dimensions",
    needsFiles: true,
  },
  {
    action: segmentByKeywords,
    hasRecursiveOption: true,
    label: "Segment by Keywords",
    needsFiles: true,
  },
  {
    action: segmentByModel,
    hasRecursiveOption: true,
    label: "Segment by Model",
    needsFiles: true,
  },
  {
    action: segmentByUpscaled,
    hasRecursiveOption: true,
    label: "Segment by Upscaled",
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
    action: generateBatches,
    hasRecursiveOption: true,
    label: "Generate Batches Exhaustively",
    needsFiles: true,
  },
  {
    action: (fileNames: FileNames) => generateBatches({ ...fileNames, shuffle: true }),
    hasRecursiveOption: true,
    label: "Generate Batches Exhaustively (Shuffled)",
    needsFiles: true,
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
        `Select an option ${chalk.grey("(with --r flag for recursion)")}: `
      )}`
    );

    const scriptIndex = +input.replace(/\D+/gi, "");

    if (isNaN(scriptIndex) || scriptIndex < 0 || scriptIndex > SCRIPT_OPTS.length)
      throw new Error(`Invalid selection. Enter a number between 0 and ${SCRIPT_OPTS.length}.`);

    if (scriptIndex === 0) return readline.close();

    const script = SCRIPT_OPTS[scriptIndex - 1];
    if (!script) throw new Error("Failed to find matching script.");

    if (script.needsFiles) {
      const withRecursion = script.hasRecursiveOption && input.includes("--r");
      const fileNames = await listImageAndParamFileNames(withRecursion);
      script.action(fileNames);
    } else script.action();
  } catch (err) {
    console.error(chalk.red(err));
    console.log(chalk.grey("-".repeat(100)));
    main();
  }
};

main();

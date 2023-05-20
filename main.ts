import chalk from "chalk";
import { prompt, readline } from "./utils";
import { FileNames } from "./types";
import {
  convertImagesInCurDirToJPG,
  generateBatch,
  generateBatchesExhaustive,
  generateImages,
  generateLoraTrainingFolderAndParams,
  generateTxt2ImgOverrides,
  getActiveModel,
  getActiveVAE,
  listImageAndParamFileNames,
  listModels,
  listVAEs,
  mainEmitter,
  pruneFilesFoundInFolders,
  pruneImageParams,
  pruneParamsAndSegmentUpscaled,
  segmentByKeywords,
  segmentByModel,
  segmentByUpscaled,
} from "./endpoints";

const SCRIPT_OPTS: {
  action: ({ imageFileNames, paramFileNames }?: FileNames) => any;
  needsFiles: boolean;
  needsRecursiveFiles?: boolean;
  label: string;
}[] = [
  {
    action: generateBatchesExhaustive,
    label: "Generate Batches Exhaustively",
    needsFiles: true,
  },
  {
    action: generateBatchesExhaustive,
    label: "Generate Batches Exhaustively (Recursive)",
    needsFiles: true,
    needsRecursiveFiles: true,
  },
  {
    action: (fileNames: FileNames) => generateBatchesExhaustive({ ...fileNames, shuffle: true }),
    label: "Generate Batches Exhaustively (Shuffled)",
    needsFiles: true,
  },
  {
    action: (fileNames: FileNames) => generateBatchesExhaustive({ ...fileNames, shuffle: true }),
    label: "Generate Batches Exhaustively (Shuffled) (Recursive)",
    needsFiles: true,
    needsRecursiveFiles: true,
  },
  {
    action: generateBatch,
    label: "Generate Batch",
    needsFiles: true,
  },
  {
    action: (fileNames: FileNames) => generateImages({ ...fileNames, mode: "upscale" }),
    label: "Upscale (Hires Fix)",
    needsFiles: true,
  },
  {
    action: (fileNames: FileNames) => generateImages({ ...fileNames, mode: "upscale" }),
    label: "Upscale (Hires Fix) (Recursive)",
    needsFiles: true,
    needsRecursiveFiles: true,
  },
  {
    action: pruneParamsAndSegmentUpscaled,
    label: "Prune Params & Segment by Upscaled",
    needsFiles: true,
  },
  {
    action: pruneImageParams,
    label: "Prune Generation Parameters",
    needsFiles: true,
  },
  {
    action: pruneImageParams,
    label: "Prune Generation Parameters (Recursive)",
    needsFiles: true,
    needsRecursiveFiles: true,
  },
  {
    action: pruneFilesFoundInFolders,
    label: "Prune Files Found in Other Folders",
    needsFiles: true,
  },
  {
    action: segmentByUpscaled,
    label: "Segment by Upscaled",
    needsFiles: true,
  },
  {
    action: segmentByKeywords,
    label: "Segment by Keywords",
    needsFiles: true,
  },
  {
    action: segmentByModel,
    label: "Segment by Model",
    needsFiles: true,
  },
  {
    action: (fileNames: FileNames) => generateImages({ ...fileNames, mode: "reproduce" }),
    label: "Segment by Reproducible",
    needsFiles: true,
  },
  {
    action: (fileNames: FileNames) => generateImages({ ...fileNames, mode: "reproduce" }),
    label: "Segment by Reproducible (Recursive)",
    needsFiles: true,
    needsRecursiveFiles: true,
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
    action: async () => {
      await getActiveModel(true);
      mainEmitter.emit("done");
    },
    label: "Get Active Model",
    needsFiles: false,
  },
  {
    action: async () => {
      await getActiveVAE(true);
      mainEmitter.emit("done");
    },
    label: "Get Active VAE",
    needsFiles: false,
  },
  {
    action: async () => {
      await listModels(true);
      mainEmitter.emit("done");
    },
    label: "List Models",
    needsFiles: false,
  },
  {
    action: async () => {
      await listVAEs(true);
      mainEmitter.emit("done");
    },
    label: "List VAEs",
    needsFiles: false,
  },
];

export const main = async () => {
  try {
    const scriptIndex = +(await prompt(
      `${SCRIPT_OPTS.map((opt, i) => `  ${chalk.cyan(i + 1)}: ${opt.label}`).join(
        "\n"
      )}\n${chalk.red("  0:")} Exit\n${chalk.blueBright("Select an option: ")} `
    ));

    if (isNaN(scriptIndex) || scriptIndex < 0 || scriptIndex > SCRIPT_OPTS.length)
      throw new Error(`Invalid selection. Enter a number between 0 and ${SCRIPT_OPTS.length}.`);

    if (scriptIndex === 0) return readline.close();

    const script = SCRIPT_OPTS[scriptIndex - 1];
    if (!script) throw new Error("Failed to find matching script.");

    if (script.needsFiles) {
      const fileNames = await listImageAndParamFileNames(script.needsRecursiveFiles);
      script.action(fileNames);
    } else script.action();
  } catch (err) {
    console.error(chalk.red(err));
  }
};

main();

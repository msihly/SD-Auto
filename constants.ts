import env from "./env";

export const API_URL = env.API_URL ?? "http://127.0.0.1:7860/sdapi/v1";

export const AUTO1111_FOLDER_CONFIG_FILE_NAME =
  env.AUTO1111_FOLDER_CONFIG_FILE_NAME ?? "folder-config.json";

export const DEFAULTS = {
  BATCH_SIZE: env.DEFAULT_BATCH_SIZE ?? 40,
  BATCH_EXCLUDED_PATHS: env.DEFAULT_BATCH_EXCLUDED_PATHS ?? [],
  HIRES_DENOISING_STRENGTH: env.DEFAULT_HIRES_DENOISING_STRENGTH ?? 0.3,
  HIRES_SCALE: env.DEFAULT_HIRES_SCALE ?? 2,
  HIRES_UPSCALER: env.DEFAULT_HIRES_UPSCALER ?? "ESRGAN_4x",
  LORA_MODEL: env.DEFAULT_LORA_MODEL,
  LORA_PARAMS_PATH: env.DEFAULT_LORA_PARAMS_PATH,
};

export const DIR_NAMES = {
  nonJPG: "Non-JPG",
  nonReproducible: "Non-Reproducible",
  nonUpscaled: "Non-Upscaled",
  products: "Products",
  prunedImagesOtherFolders: "Pruned Images (Other Folders)",
  prunedParams: "Pruned Params",
  reproducible: "Reproducible",
  upscaleCompleted: "Upscale Completed",
  upscaled: "Upscaled",
  ...(env.DIR_NAMES ?? {}),
};

export const EXTS = {
  IMAGE: "jpg",
  PARAMS: "txt",
};

export const LORA = {
  INPUT_DIR: "input",
  LOGS_DIR: "logs",
  OUTPUT_DIR: "output",
  TRAINING_PARAMS_FILE_NAME: env.LORA_TRAINING_PARAMS_FILE_NAME ?? "training-params.json",
};

export const OUTPUT_DIR = env.OUTPUT_DIR;

export const REPROD_DIFF_TOLERANCE = env.REPROD_DIFF_TOLERANCE ?? 0.15;

export const TXT2IMG_OVERRIDES_FILE_NAME =
  env.TXT2IMG_OVERRIDES_FILE_NAME ?? "txt2img-overrides.json";

export const VAE_DIR = env.VAE_DIR;

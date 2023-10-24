import env from "./env";

export const API_URL = env.API_URL ?? "http://127.0.0.1:7860/sdapi/v1";

export const AUTO1111_FOLDER_CONFIG_FILE_NAME =
  env.AUTO1111_FOLDER_CONFIG_FILE_NAME ?? "folder-config.json";

export const DEFAULTS_BASE = {
  ADETAILER_MODEL: "face_yolov8n.pt",
  BATCH_SIZE: 40,
  BATCH_EXCLUDED_PATHS: [],
  HIRES_DENOISING_STRENGTH: 0.3,
  HIRES_SCALE: 2,
  HIRES_UPSCALER: "ESRGAN_4x",
  LORA_MODEL: null,
  LORA_PARAMS_PATH: null,
  RESTORE_FACES_STRENGTH: 0.4,
};

export const DEFAULTS = { ...DEFAULTS_BASE, ...(env.DEFAULTS ?? {}) };

export const DIR_NAMES_BASE = {
  EMPTY: "Empty Folders",
  NON_JPG: "Non-JPG",
  NON_REPRODUCIBLE: "Non-Reproducible",
  NON_UPSCALED: "Non-Upscaled",
  PRODUCTS: "Products",
  PRUNED_IMAGES: "Pruned Images",
  PRUNED_IMAGES_OTHER_FOLDERS: "Pruned Images (Other Folders)",
  PRUNED_PARAMS: "Pruned Params",
  REPRODUCIBLE: "Reproducible",
  RESTORED_FACES: "Restored Faces",
  UPSCALED: "Upscaled",
};

export const DIR_NAMES = { ...DIR_NAMES_BASE, ...(env.DIR_NAMES ?? {}) };

export const EXTS = {
  IMAGE: "jpg",
  PARAMS: "txt",
};

export const FILE_NAMES = {
  ORIGINAL: "Original",
  REPRODUCED: "Reproduced",
  RESTORED_FACES: "Restored Faces",
  UPSCALED: "Upscaled",
};

export const FILE_AND_DIR_NAMES = [...Object.values(DIR_NAMES), ...Object.values(FILE_NAMES)];

export const LORA = {
  INPUT_DIR: "input",
  LOGS_DIR: "logs",
  OUTPUT_DIR: "output",
  TRAINING_PARAMS_FILE_NAME: env.LORA_TRAINING_PARAMS_FILE_NAME ?? "training-params.json",
};

export const OUTPUT_DIR = env.OUTPUT_DIR;

export const REPROD_DIFF_TOLERANCE = env.REPROD_DIFF_TOLERANCE ?? 0.15;

export const RESTORE_FACES_TYPES = {
  ADETAILER: "ADetailer",
  CODEFORMER: "CodeFormer",
  GFPGAN: "GFPGAN",
} as const;
export type RestoreFacesType = (typeof RESTORE_FACES_TYPES)[keyof typeof RESTORE_FACES_TYPES];

export const TXT2IMG_OVERRIDES_FILE_NAME =
  env.TXT2IMG_OVERRIDES_FILE_NAME ?? "txt2img-overrides.json";

export const VAE_DIR = env.VAE_DIR;

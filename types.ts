export type ImageFileNames = { imageFileNames: string[] };
export type ParamFileNames = { paramFileNames: string[] };
export type FileNames = ImageFileNames & ParamFileNames;

export type Auto1111FolderConfig = {
  controlNetPoses?: string;
  embeddings?: string;
  extensions?: string;
  lora?: string;
  lycoris?: string;
  models?: string;
  outputs?: string;
  vae?: string;
};
export type Auto1111FolderName = keyof Auto1111FolderConfig;

export type ExtrasRequest = {
  codeformer_visibility?: number;
  codeformer_weight?: number;
  extras_upscaler_2_visibility?: number;
  gfpgan_visibility?: number;
  image: string;
  resize_mode?: 0 | 1;
  show_extras_results?: boolean;
  upscale_first?: boolean;
  upscaler_1?: string;
  upscaler_2?: string;
  upscaling_crop?: boolean;
  upscaling_resize?: number;
  upscaling_resize_h?: number;
  upscaling_resize_w?: number;
};

export type ImageParams = {
  cfgScale?: number;
  clipSkip?: number;
  cutoffDisableForNeg?: boolean;
  cutoffEnabled?: boolean;
  cutoffInterpolation?: string;
  cutoffPadding?: string;
  cutoffStrong?: boolean;
  cutoffTargets?: string[];
  cutoffWeight?: number;
  fileName?: string;
  height?: number;
  hiresDenoisingStrength?: number;
  hiresScale?: number;
  hiresSteps?: number;
  hiresUpscaler?: string;
  model?: string;
  negPrompt?: string;
  negTemplate?: string;
  prompt?: string;
  rawParams?: string;
  restoreFaces?: boolean;
  restoreFacesStrength?: number;
  sampler?: string;
  seed?: number;
  steps?: number;
  subseed?: number;
  subseedStrength?: number;
  template?: string;
  tiledDiffusion?: {
    batchSize: number;
    keepInputSize: "True" | "False";
    method: "MultiDiffusion" | "Mixture of Diffusers";
    overwriteSize: "True" | "False";
    tileHeight: number;
    tileOverlap: number;
    tileWidth: number;
  };
  tiledVAE?: {
    colorFixEnabled: "True" | "False";
    decoderTileSize: number;
    encoderTileSize: number;
    fastDecoderEnabled: "True" | "False";
    fastEncoderEnabled: "True" | "False";
    vaeToGPU: "True" | "False";
  };
  vae?: string;
  vaeHash?: string;
  width?: number;
};
export type Txt2ImgOverrides = Omit<ImageParams, "fileName" | "rawParams" | "vaeHash">;
export type Txt2ImgOverride = keyof Txt2ImgOverrides;

export type LoraTrainingParams = {
  cache_latents?: boolean;
  caption_extension?: string;
  clip_skip?: number;
  color_aug?: boolean;
  enable_bucket?: boolean;
  epoch?: string;
  flip_aug?: boolean;
  full_fp16?: boolean;
  gradient_accumulation_steps?: number;
  gradient_checkpointing?: boolean;
  keep_tokens?: string;
  learning_rate?: string;
  logging_dir?: string;
  lora_network_weights?: string;
  lr_scheduler?: string;
  lr_scheduler_num_cycles?: string;
  lr_scheduler_power?: string;
  lr_warmup?: string;
  max_data_loader_n_workers?: string;
  max_resolution?: string;
  max_token_length?: string;
  max_train_epochs?: string;
  mem_eff_attn?: boolean;
  mixed_precision?: string;
  model_list?: string;
  network_alpha?: number;
  network_dim?: number;
  no_token_padding?: boolean;
  num_cpu_threads_per_process?: number;
  output_dir?: string;
  output_name?: string;
  pretrained_model_name_or_path?: string;
  prior_loss_weight?: number;
  reg_data_dir?: string;
  resume?: string;
  save_every_n_epochs?: string;
  save_model_as?: string;
  save_precision?: string;
  save_state?: boolean;
  seed?: string;
  shuffle_caption?: boolean;
  stop_text_encoder_training?: number;
  text_encoder_lr?: string;
  train_batch_size?: number;
  train_data_dir?: string;
  training_comment?: string;
  unet_lr?: string;
  use_8bit_adam?: boolean;
  v2?: boolean;
  v_parameterization?: boolean;
  xformers?: boolean;
};

export type Model = { hash: string; name: string; path: string };

export type NoEmit = { noEmit?: boolean };

export type Txt2ImgMode = "reproduce" | "upscale";

export type Txt2ImgOverrideGroup = {
  dirPath: string;
  fileNames: string[];
  overrides?: Txt2ImgOverrides;
};

export type Txt2ImgRequest = {
  alwayson_scripts?: object;
  cfg_scale: number;
  denoising_strength?: number;
  enable_hr: boolean;
  height: number;
  hr_scale?: number;
  hr_steps?: number;
  hr_upscaler?: string;
  negative_prompt: string;
  override_settings?: object;
  override_settings_restore_afterwards?: boolean;
  prompt: string;
  restore_faces?: boolean;
  sampler_name: string;
  save_images: boolean;
  seed: number;
  send_images: boolean;
  steps: number;
  subseed?: number;
  subseed_strength?: number;
  width: number;
};

export type VAE = {
  fileName: string;
  hash: string;
};

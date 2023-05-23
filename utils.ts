import fs from "fs/promises";
import { createReadStream } from "fs";
import { BinaryToTextEncoding, createHash } from "crypto";
import { createInterface } from "readline";
import chalk from "chalk";
import path from "path";
import md5File from "md5-file";
import sharp from "sharp";
import pixelmatch from "pixelmatch";
import dayjs from "dayjs";
import dayjsPluginDuration from "dayjs/plugin/duration.js";

export type Image = {
  ext: string;
  hashMD5: string;
  path: string;
  name: string;
};

export class PromiseQueue {
  queue = Promise.resolve();

  add(fn) {
    return new Promise((resolve, reject) => {
      this.queue = this.queue.then(fn).then(resolve).catch(reject);
    });
  }
}

dayjs.extend(dayjsPluginDuration);
export { dayjs };

export const readline = createInterface({ input: process.stdin, output: process.stdout });

export const chunkArray = <T>(arr: T[], size: number): T[][] =>
  [...Array(Math.ceil(arr.length / size))].map((_, i) => arr.slice(i * size, i * size + size));

export const compareImage = async (imgInput1: string | Buffer, imgInput2: string | Buffer) => {
  const [img1, img2] = await Promise.all(
    [imgInput1, imgInput2].map((imgInput) =>
      sharp(imgInput).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
    )
  );

  const { width, height } = img1.info;

  const pixelDiff = pixelmatch(img1.data, img2.data, null, width, height, {
    threshold: 0.1,
  });
  const percentDiff = pixelDiff / (width * height);

  return { pixelDiff, percentDiff };
};

export const convertImagesToJPG = async (imagePaths: string[]) => {
  return await Promise.all(
    imagePaths.map((imagePath) => {
      try {
        const extName = path.extname(imagePath);
        return extName !== ".jpg"
          ? sharp(imagePath).toFile(`${path.basename(imagePath, extName)}.jpg`)
          : true;
      } catch (err) {
        console.error(chalk.red(`Error converting ${chalk.magenta(imagePath)} to JPG: ${err}`));
      }
    })
  );
};

export const delimit = (str: string, delimeter: string) =>
  str
    .replace(new RegExp(`^(${delimeter}|\s)|(${delimeter}|\s)$`, "gim"), "")
    .split(delimeter)
    .map((p) => p.trim())
    .filter((p) => p.length);

export const dirToFilePaths = async (dirPath: string): Promise<string[]> => {
  const paths = await fs.readdir(dirPath, { withFileTypes: true });
  return (
    await Promise.all(
      paths.map(async (dirent) => {
        const filePath = path.join(dirPath, dirent.name);
        return dirent.isDirectory() ? await dirToFilePaths(filePath) : filePath;
      })
    )
  ).flat();
};

export const doesPathExist = async (fileOrDirPath: string) => {
  try {
    await fs.access(fileOrDirPath);
    return true;
  } catch (err) {
    return false;
  }
};

export const escapeRegEx = (str: string) => str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export const getImagesInFolders = async ({
  paths,
  withConsole = false,
}: {
  paths: string[];
  withConsole?: boolean;
}): Promise<Image[]> => {
  if (withConsole)
    console.log(
      `Searching for images in ["${chalk.cyan(`${paths.join(`"${chalk.white(",")} "`)}`)}"]...`
    );

  const imagesInFolders = (
    await Promise.all(
      paths.map(async (p) => {
        const filePaths = await dirToFilePaths(p);
        return await Promise.all(
          filePaths.map(async (p) => ({
            ext: path.extname(p).replace(".", ""),
            hashMD5: await md5File(p),
            path: p,
            name: path.basename(p, path.extname(p)),
          }))
        );
      })
    )
  ).flat();

  if (withConsole)
    console.log(
      `Images in folders:\n${makeConsoleList(
        imagesInFolders.map(
          (img) => `Name: ${img.name}. Ext: ${img.ext}. Hash: ${img.hashMD5}. Path: ${img.path}.`
        )
      )}`
    );

  return imagesInFolders;
};

export const makeConsoleList = (items: string[], numerical = false) =>
  items
    .map((o, i) => `  ${chalk.blueBright(numerical ? `${i + 1}.` : "•")} ${chalk.white(o)}`)
    .join("\n") + "\n";

export const prompt = (query: string, callback?: (answer: string) => void): Promise<string> =>
  new Promise((resolve) =>
    readline.question(query, (answer) => {
      callback?.(answer);
      resolve(answer);
    })
  );

export const randomSort = <T>(arr: T[]): T[] => {
  const newArr = arr.slice();
  for (let i = newArr.length - 1; i > 0; i--) {
    const rand = Math.floor(Math.random() * (i + 1));
    [newArr[i], newArr[rand]] = [newArr[rand], newArr[i]];
  }
  return newArr;
};

export const round = (num: number, decimals = 2) => {
  const n = Math.pow(10, decimals);
  return Math.round((num + Number.EPSILON) * n) / n;
};

export const sha256File = async (
  filename: string,
  {
    encoding = "hex",
    length,
    offset,
  }: { encoding?: BinaryToTextEncoding; length?: number; offset?: number }
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const fileStream = createReadStream(filename, { highWaterMark: offset });
    const hash = createHash("sha256");

    fileStream.on("error", reject);

    fileStream.on("end", () => {
      const digest = hash.digest(encoding);
      resolve(digest.substring(0, length));
    });

    fileStream.on("data", (chunk: Buffer) => {
      hash.update(chunk);
    });

    fileStream.read(offset);
  });
};

import fs from "fs/promises";
import { constants as fsc, createReadStream } from "fs";
import { inspect } from "util";
import { BinaryToTextEncoding, createHash } from "crypto";
import { createInterface } from "readline";
import chalk from "chalk";
import path from "path";
import md5File from "md5-file";
import sharp from "sharp";
import pixelmatch from "pixelmatch";
import dayjs from "dayjs";
import dayjsPluginDuration from "dayjs/plugin/duration.js";

/* -------------------------------------------------------------------------- */
/*                                    TYPES                                   */
/* -------------------------------------------------------------------------- */
export type Image = {
  ext: string;
  hashMD5: string;
  path: string;
  name: string;
};

export type TreeNode = {
  children: TreeNode[];
  name: string;
};

export type NestedKeyOf<T extends object> = {
  [Key in keyof T & (string | number)]: T[Key] extends object
    ? `${Key}` | `${Key}.${NestedKeyOf<T[Key]>}`
    : `${Key}`;
}[keyof T & (string | number)];

/* -------------------------------------------------------------------------- */
/*                             CLASSES / INSTANCES                            */
/* -------------------------------------------------------------------------- */
export class PromiseQueue {
  queue = Promise.resolve();

  add(fn) {
    return new Promise((resolve, reject) => {
      this.queue = this.queue.then(fn).then(resolve).catch(reject);
    });
  }

  isPending() {
    return inspect(this.queue).includes("pending");
  }
}

dayjs.extend(dayjsPluginDuration);
export { dayjs };

export const readline = createInterface({ input: process.stdin, output: process.stdout });

/* -------------------------------------------------------------------------- */
/*                                  FUNCTIONS                                 */
/* -------------------------------------------------------------------------- */

/* --------------------------------- ARRAYS --------------------------------- */
export const chunkArray = <T>(arr: T[], size: number): T[][] =>
  [...Array(Math.ceil(arr.length / size))].map((_, i) => arr.slice(i * size, i * size + size));

export const randomSort = <T>(arr: T[]): T[] => {
  const newArr = arr.slice();
  for (let i = newArr.length - 1; i > 0; i--) {
    const rand = Math.floor(Math.random() * (i + 1));
    [newArr[i], newArr[rand]] = [newArr[rand], newArr[i]];
  }
  return newArr;
};

/* ----------------------------------- CLI ---------------------------------- */
export const makeConsoleList = (items: string[], numerical = false) =>
  items
    .map((o, i) => `  ${chalk.blueBright(numerical ? `${i + 1}.` : "•")} ${chalk.white(o)}`)
    .join("\n") + "\n";

export const makeExistingValueLog = (val: string) =>
  `${val ? ` ${chalk.grey(`(${val})`)}` : ""}${chalk.blueBright(": ")}`;

export const makeTimeLog = (time: number) =>
  `${chalk.green(dayjs.duration(time).format("H[h]m[m]s[s]"))} ${chalk.grey(`(${round(time)}ms)`)}`;

export const prompt = (query: string, callback?: (answer: string) => void): Promise<string> =>
  new Promise((resolve) =>
    readline.question(query, (answer) => {
      callback?.(answer);
      resolve(answer);
    })
  );

export const valsToOpts = (vals: string[]) => vals.map((v) => ({ label: v, value: v }));

/* ---------------------------------- FILES --------------------------------- */
export const dirToFilePaths = async (
  dirPath: string,
  recursive: boolean = true
): Promise<string[]> => {
  const paths = await fs.readdir(dirPath, { withFileTypes: true });
  return (
    await Promise.all(
      paths.map(async (dirent) => {
        const filePath = path.join(dirPath, dirent.name);
        return dirent.isDirectory()
          ? recursive
            ? await dirToFilePaths(filePath)
            : null
          : filePath;
      })
    )
  )
    .flat()
    .filter((p) => p !== null);
};

export const doesPathExist = async (fileOrDirPath: string) => {
  try {
    await fs.access(fileOrDirPath);
    return true;
  } catch (err) {
    return false;
  }
};

export const extendFileName = (fileName: string, ext: string) =>
  `${path.relative(".", fileName).replace(/\.\w+$/, "")}.${ext}`;

export const listFolders = async (dirPath: string = "."): Promise<string[]> => {
  const paths = await fs.readdir(dirPath, { withFileTypes: true });
  if (paths.length === 0) return [];
  console.log("Paths", paths);

  return (
    await Promise.all(
      paths.map(async (dirent) => {
        if (dirent.isDirectory()) {
          const subDirPath = path.join(dirPath, dirent.name);
          const subPaths = await fs.readdir(subDirPath, { withFileTypes: true });
          const subDirs = subPaths.filter((d) => d.isDirectory());
          return subDirs.length ? await listFolders(subDirPath) : subDirPath;
        }
      })
    )
  )
    .flat()
    .filter((d) => d);
};

export const listEmptyFolders = async (dirPath: string = ".") => {
  const folders = await listFolders(dirPath);
  return (
    await Promise.all(
      folders.map(async (folder) => ((await fs.readdir(folder)).length === 0 ? folder : undefined))
    )
  ).filter((f) => f);
};

export const makeDirs = async (dirNames: string[]) =>
  Promise.all(dirNames.map((dirName) => fs.mkdir(dirName, { recursive: true })));

export const makeFilePathInDir = (dirName: string, fileName: string) =>
  path.join(dirName, path.relative(".", fileName));

export const moveFile = async (
  filePath: string,
  folderName: string,
  newName?: string
): Promise<string> => {
  const newFilePath = makeFilePathInDir(folderName, newName ?? filePath);
  if (filePath === newFilePath) return newFilePath;

  try {
    await makeDirs([folderName, path.dirname(newFilePath)]);
    await fs.copyFile(filePath, newFilePath, fsc.COPYFILE_EXCL);
    await fs.unlink(filePath);
    return newFilePath;
  } catch (err) {
    if (err.code === "EEXIST") {
      const suffixedPath = (() => {
        const pathParts = path.parse(newFilePath);
        const match = pathParts.name.match(/^(.+)\s\((\d+)\)$/);
        if (match) {
          const [, name, num] = match;
          return `${name} (${+num + 1})${pathParts.ext}`;
        } else return `${pathParts.name} (1)${pathParts.ext}`;
      })();

      console.log(chalk.yellow(`Renaming to ${chalk.cyan(suffixedPath)}.`));
      return await moveFile(filePath, folderName, suffixedPath);
    }

    if (err.code === "ENOENT")
      console.warn(chalk.yellow(`File ${chalk.cyan(filePath)} not found.`));
    else console.error(chalk.red(err.stack));

    return null;
  }
};

export const moveFilesToFolder = async ({
  exts,
  filePath,
  folderColor = "magenta",
  folderName,
  newName,
  withConsole = false,
}: {
  exts?: string[];
  filePath: string;
  folderColor?: "cyan" | "green" | "magenta" | "red" | "yellow";
  folderName: string;
  newName?: string;
  withConsole?: boolean;
}) => {
  try {
    if (!exts?.length) await moveFile(filePath, folderName, newName);
    else {
      await Promise.all(
        exts.map(async (ext) => {
          const originalName = extendFileName(filePath, ext);
          const fileName = newName ? extendFileName(newName, ext) : originalName;
          return moveFile(originalName, folderName, fileName);
        })
      );
    }

    if (withConsole)
      console.log(`Moved ${chalk.cyan(filePath)} to ${chalk[folderColor](folderName)}.`);
  } catch (err) {
    console.error(chalk.red(err.stack));
  }
};

export const removeEmptyFolders = async (dirPath: string = ".", excluded?: string[]) => {
  if (!(await fs.stat(dirPath)).isDirectory() || excluded?.includes(path.basename(dirPath))) return;

  let files = await fs.readdir(dirPath);
  if (files.length) {
    await Promise.all(files.map((f) => removeEmptyFolders(path.join(dirPath, f), excluded)));
    files = await fs.readdir(dirPath);
  }

  if (!files.length && path.resolve(dirPath) !== path.resolve(process.cwd()))
    await fs.rmdir(dirPath);
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

export const writeFilesToFolder = async ({
  files,
  folderName,
}: {
  files: { content: string | Buffer; ext: string; name: string }[];
  folderName: string;
}) => {
  try {
    await makeDirs([folderName]);

    await Promise.all(
      files.map((file) =>
        fs.writeFile(
          makeFilePathInDir(folderName, extendFileName(file.name, file.ext)),
          file.content
        )
      )
    );
  } catch (err) {
    console.error(chalk.red(err.stack));
  }
};

/* --------------------------------- IMAGES --------------------------------- */
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

export const listImagesFoundInOtherFolders = ({
  imageFileNames,
  imagesInOtherFolders,
  withConsole = false,
}: {
  imageFileNames: string[];
  imagesInOtherFolders: Image[];
  withConsole?: boolean;
}) => {
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
/* ------------------------------ MISCELLANEOUS ----------------------------- */
const createTreeNode = (dirPath: string, tree: TreeNode[]) => {
  const dirNames = path.normalize(dirPath).split(path.sep) as string[];
  const [rootDirName, ...remainingDirNames] = dirNames;
  const treeNode = tree.find((t) => t.name === rootDirName);
  if (!treeNode) tree.push({ name: rootDirName, children: [] });
  if (remainingDirNames.length > 0)
    createTreeNode(path.join(...remainingDirNames), (treeNode ?? tree[tree.length - 1]).children);
};

export const createTree = (paths: string[]): TreeNode[] =>
  paths.reduce((acc, cur) => (createTreeNode(cur, acc), acc), []);

export const delimit = (str: string, delimeter: string) =>
  str
    .replace(new RegExp(`^(${delimeter}|\s)|(${delimeter}|\s)$`, "gim"), "")
    .split(delimeter)
    .map((p) => p.trim())
    .filter((p) => p.length);

export const dotKeysToTree = <T>(obj: T): T =>
  Object.entries(obj).reduce((acc, [key, value]) => {
    const keys = key.split(".");
    const lastKey = keys.pop()!;
    return (keys.reduce((obj, k) => (obj[k] ||= {}), acc)[lastKey] = value), acc;
  }, {} as T);

export const escapeRegEx = (str: string) => str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/* ---------------------------------- MATH ---------------------------------- */
export const round = (num: number, decimals = 2) => {
  const n = Math.pow(10, decimals);
  return Math.round((num + Number.EPSILON) * n) / n;
};

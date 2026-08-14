'use strict';

import fs from 'fs';
import path from 'path';

/**
 * サンプル実装に置いたノートブックを、ドキュメントサイトへ取り込む。
 *
 * ノートブックの正本は apps/ 配下にある。実装本体（src / build）を相対パスで
 * 読み込むため、実装の隣に置く必要があるからだ。MkDocs は docs_dir の外を
 * 参照できないので、ビルド前にここへ複製する。複製先は生成物として
 * .gitignore に入れてある。
 */
const NOTEBOOK_SOURCES = [
  { app: 'apps/grokking-ml-python/notebooks', docs: 'docs/article/grokking-machine-learning/python/notebooks' },
  { app: 'apps/grokking-ml-kotlin/notebooks', docs: 'docs/article/grokking-machine-learning/kotlin/notebooks' },
  { app: 'apps/grokking-ml-fsharp/notebooks', docs: 'docs/article/grokking-machine-learning/fsharp/notebooks' },
  // ライブラリ版（原著ノートブックの再現）
  { app: 'apps/grokking-ml-python-lib/notebooks', docs: 'docs/article/grokking-machine-learning/lib/python/notebooks' },
  { app: 'apps/grokking-ml-kotlin-lib/notebooks', docs: 'docs/article/grokking-machine-learning/lib/kotlin/notebooks' },
  { app: 'apps/grokking-ml-fsharp-lib/notebooks', docs: 'docs/article/grokking-machine-learning/lib/fsharp/notebooks' },
];

/**
 * ノートブックタスクを gulp に登録する
 * @param {import('gulp').Gulp} gulp - Gulp インスタンス
 */
export default function (gulp) {
  gulp.task('notebooks:sync', (done) => {
    try {
      let copied = 0;
      for (const { app, docs } of NOTEBOOK_SOURCES) {
        const source = path.join(process.cwd(), app);
        const target = path.join(process.cwd(), docs);
        if (!fs.existsSync(source)) {
          console.warn(`Warning: ${app} が見つかりません。スキップします。`);
          continue;
        }
        fs.mkdirSync(target, { recursive: true });
        for (const name of fs.readdirSync(source)) {
          if (!name.endsWith('.ipynb')) continue;
          fs.copyFileSync(path.join(source, name), path.join(target, name));
          copied += 1;
        }
      }
      console.log(`Synced ${copied} notebooks into docs/.`);
      done();
    } catch (error) {
      done(error);
    }
  });

  gulp.task('notebooks:check', (done) => {
    try {
      const problems = [];
      let checked = 0;
      for (const { app } of NOTEBOOK_SOURCES) {
        const source = path.join(process.cwd(), app);
        if (!fs.existsSync(source)) continue;
        for (const name of fs.readdirSync(source).filter((f) => f.endsWith('.ipynb'))) {
          const file = path.join(source, name);
          const notebook = JSON.parse(fs.readFileSync(file, 'utf8'));
          const codeCells = notebook.cells.filter((c) => c.cell_type === 'code');
          const unexecuted = codeCells.filter((c) => !c.execution_count);
          const errored = codeCells.filter((c) =>
            (c.outputs || []).some((o) => o.output_type === 'error'),
          );
          if (unexecuted.length > 0) {
            problems.push(`${app}/${name}: 未実行のセルが ${unexecuted.length} 個あります`);
          }
          if (errored.length > 0) {
            problems.push(`${app}/${name}: エラー出力のセルが ${errored.length} 個あります`);
          }
          checked += 1;
        }
      }
      if (problems.length > 0) {
        problems.forEach((p) => console.error(`NG ${p}`));
        done(new Error('実行済みでないノートブックがあります。カーネルで再実行してください。'));
        return;
      }
      console.log(`Checked ${checked} notebooks: すべて実行済み・エラーなし。`);
      done();
    } catch (error) {
      done(error);
    }
  });

  gulp.task('notebooks:clean', (done) => {
    try {
      for (const { docs } of NOTEBOOK_SOURCES) {
        const target = path.join(process.cwd(), docs);
        if (fs.existsSync(target)) {
          fs.rmSync(target, { recursive: true, force: true });
        }
      }
      console.log('Removed synced notebooks from docs/.');
      done();
    } catch (error) {
      done(error);
    }
  });
}

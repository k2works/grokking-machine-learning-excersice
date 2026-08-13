'use strict';

/**
 * Gulpfile that loads tasks from the script directory
 */

import 'dotenv/config';
import gulp from 'gulp';
import mkdocsTasks from './ops/scripts/mkdocs.js';
import manualTasks from './ops/scripts/manual.js';
import journalTasks from './ops/scripts/journal.js';
import notebookTasks from './ops/scripts/notebooks.js';
import vaultTasks from './ops/scripts/vault.js';
import sshTasks from './ops/scripts/ssh.js';
import sonarLocalTasks from './ops/scripts/sonar_local.js';

// Load gulp tasks from script modules
mkdocsTasks(gulp);
manualTasks(gulp);
journalTasks(gulp);
notebookTasks(gulp);
vaultTasks(gulp);
sshTasks(gulp);
sonarLocalTasks(gulp);

// ノートブックを取り込んでからドキュメントを扱う
export const docsBuild = gulp.series('notebooks:sync', 'mkdocs:build');
export const docsServe = gulp.series('notebooks:sync', 'mkdocs:serve');

export const spec = gulp.series('notebooks:sync', 'mkdocs:serve', 'mkdocs:open');

// Export gulp to make it available to the gulp CLI
export default gulp;

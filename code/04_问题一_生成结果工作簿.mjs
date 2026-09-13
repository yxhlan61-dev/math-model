import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve(import.meta.dirname, "..");
const outputDir = path.join(root, "code", "outputs", "问题一");
const intermediateDir = path.join(root, "tmp", "问题一_求解中间结果");
const previewDir = path.join(root, "tmp", "问题一_结果工作簿预览");
const templatePath = path.join(root, "附件", "附件5", "result1.xlsx");
const artifactToolEntry = process.env.CODEX_ARTIFACT_TOOL_PATH ?? path.join(
  os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies",
  "node", "node_modules", "@oai", "artifact-tool", "dist", "artifact_tool.mjs",
);
const { FileBlob, SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactToolEntry).href);

const FONT = "Arial";
const HEADER = "#1F4E78";
const methods = ["LP", "MILP", "无储能基准"];
const checkLabels = {
  feasible: "全部硬约束是否通过",
  max_abs_power_balance_residual_kwh: "最大电力平衡残差(kWh)",
  max_abs_soc_recursion_residual_kwh: "最大储电递推残差(kWh)",
  min_soc_kwh: "最低储电量(kWh)",
  max_soc_kwh: "最高储电量(kWh)",
  terminal_soc_kwh: "日末储电量(kWh)",
  max_charge_kwh_per_interval: "单时段最大充电量(kWh)",
  max_discharge_kwh_per_interval: "单时段最大放电量(kWh)",
  simultaneous_charge_discharge_count: "同时充放电时段数",
  simultaneous_interval_ids: "同时充放电时段编号",
  max_min_charge_discharge_kwh: "同时充放电重叠量最大值(kWh)",
  objective_yuan: "求解器目标值(元)",
  recalculated_cost_yuan: "独立复算购电费用(元)",
  objective_recalculation_difference_yuan: "目标值与复算费用之差(元)",
  total_purchase_kwh: "全天购电量(kWh)",
  total_charge_kwh: "全天充电量(kWh)",
  total_discharge_kwh: "全天放电量(kWh)",
  total_pv_curtailment_kwh: "全天弃光量(kWh)",
  solve_seconds: "求解耗时(s)",
  solver_status: "求解器状态码",
  solver_message: "求解器状态说明",
  solver_details: "求解器详细信息",
};

function excelColumnName(indexOneBased) {
  let n = indexOneBased;
  let result = "";
  while (n > 0) {
    n -= 1;
    result = String.fromCharCode(65 + (n % 26)) + result;
    n = Math.floor(n / 26);
  }
  return result;
}

function scalar(value) {
  if (Array.isArray(value) || (value && typeof value === "object")) return JSON.stringify(value);
  return value;
}

function styleTitle(sheet, range) {
  sheet.getRange(range).format = { font: { name: FONT, size: 14, bold: true, color: "#222222" } };
}

function styleHeader(sheet, range) {
  sheet.getRange(range).format = {
    fill: HEADER,
    font: { name: FONT, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
  };
}

function formatResultWorkbook(workbook, bundle) {
  const summary = workbook.worksheets.add("结果摘要");
  const detail = workbook.worksheets.add("逐时段结果");
  summary.showGridLines = false;
  detail.showGridLines = false;
  summary.tabColor = HEADER;
  detail.tabColor = "#5B9BD5";

  summary.getRange("A2").values = [[`问题一 ${bundle.method} 求解结果`]];
  styleTitle(summary, "A2:D2");
  summary.getRange("A4:B4").values = [["校验指标", "结果"]];
  styleHeader(summary, "A4:B4");
  const checkRows = Object.entries(bundle.checks).map(([key, value]) => [checkLabels[key] ?? key, scalar(value)]);
  summary.getRange("A5").write(checkRows);
  const fourStart = 6 + checkRows.length;
  summary.getRange(`A${fourStart}`).values = [["四小时调度汇总"]];
  summary.getRange(`A${fourStart}`).format = { font: { name: FONT, size: 11, bold: true } };
  summary.getRange(`A${fourStart + 1}`).write([bundle.fourHourColumns, ...bundle.fourHourRows]);
  styleHeader(summary, `A${fourStart + 1}:E${fourStart + 1}`);
  summary.getRange(`B5:B${fourStart - 2}`).format.numberFormat = "#,##0.000000";
  summary.getRange(`B${fourStart + 2}:E${fourStart + 7}`).format.numberFormat = "#,##0.000000";
  summary.getRange(`A4:E${fourStart + 7}`).format.font.name = FONT;
  summary.getRange("A:A").format.columnWidth = 43;
  summary.getRange("B:B").format.columnWidth = 25;
  summary.getRange("C:E").format.columnWidth = 20;

  detail.getRange("A2").values = [[`问题一 ${bundle.method} 逐时段优化结果`]];
  styleTitle(detail, "A2:S2");
  detail.getRange("A4").write([bundle.detailColumns, ...bundle.detailRows]);
  const lastCol = excelColumnName(bundle.detailColumns.length);
  styleHeader(detail, `A4:${lastCol}4`);
  detail.getRange(`A4:${lastCol}${4 + bundle.detailRows.length}`).format.font = { name: FONT, size: 9, color: "#222222" };
  detail.getRange("E:R").format.numberFormat = "#,##0.000000";
  detail.getRange("A:A").format.columnWidth = 10;
  detail.getRange("B:D").format.columnWidth = 18;
  detail.getRange(`E:${lastCol}`).format.columnWidth = 18;
  detail.freezePanes.freezeRows(4);
  detail.freezePanes.freezeColumns(4);
}

async function exportAndVerify(workbook, outputPath, previewStem) {
  workbook.recalculate();
  const inspection = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 3000 });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 100 },
    summary: "最终公式错误扫描",
  });
  for (const sheet of workbook.worksheets.items) {
    const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
    const safeName = sheet.name.replace(/[\\/:*?"<>|]/g, "_");
    await fs.writeFile(path.join(previewDir, `${previewStem}_${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  const blob = await SpreadsheetFile.exportXlsx(workbook);
  let preservedExisting = false;
  try {
    await blob.save(outputPath);
  } catch (error) {
    if (error?.code !== "EBUSY") throw error;
    // Excel 中已打开的结果文件会被 Windows 锁定。保留此前已验证版本并继续生成其他文件。
    await fs.access(outputPath);
    preservedExisting = true;
    console.warn(`文件被占用，保留已有版本：${outputPath}`);
  }
  return { outputPath, preservedExisting, inspection: inspection.ndjson, formulaErrors: errors.ndjson };
}

async function makeSubmission(bundle, method) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
  const purchaseSheet = workbook.worksheets.getItemAt(0);
  const storageSheet = workbook.worksheets.getItemAt(1);
  purchaseSheet.getRange("B2:B145").values = bundle.submission.purchase.map((value) => [value]);
  storageSheet.getRange("B2:C7").values = bundle.submission.fourHourCharge.map((value, index) => [value, bundle.submission.fourHourDischarge[index]]);
  storageSheet.getRange("E2:E3").values = [[bundle.submission.initialSoc], [bundle.submission.terminalSoc]];
  purchaseSheet.getRange("B2:B145").format.numberFormat = "#,##0.000000";
  storageSheet.getRange("B2:C7").format.numberFormat = "#,##0.000000";
  storageSheet.getRange("E2:E3").format.numberFormat = "#,##0.000000";
  const outputName = method === "无储能基准"
    ? "问题一_无储能基准_result1提交格式.xlsx"
    : `问题一_${method}_求解结果_result1提交格式.xlsx`;
  return exportAndVerify(workbook, path.join(outputDir, outputName), `${method}_提交模板`);
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const records = [];
for (const method of methods) {
  const inputName = method === "无储能基准"
    ? "问题一_无储能基准_工作簿数据.json"
    : `问题一_${method}_求解结果_工作簿数据.json`;
  const bundle = JSON.parse(await fs.readFile(path.join(intermediateDir, inputName), "utf8"));
  records.push(await makeSubmission(bundle, method));
}

const compareData = JSON.parse(await fs.readFile(path.join(intermediateDir, "问题一_三种方案结果对比.json"), "utf8"));
const compareWorkbook = Workbook.create();
const compareSheet = compareWorkbook.worksheets.add("核心结果对比");
compareSheet.showGridLines = false;
compareSheet.tabColor = HEADER;
compareSheet.getRange("A2").values = [["问题一无储能、LP 与 MILP 结果对比"]];
styleTitle(compareSheet, "A2:K2");
compareSheet.getRange("A4").write([compareData.columns, ...compareData.rows]);
const compareLastCol = excelColumnName(compareData.columns.length);
styleHeader(compareSheet, `A4:${compareLastCol}4`);
compareSheet.getRange(`B5:${compareLastCol}7`).format.numberFormat = "#,##0.000000";
compareSheet.getRange("A:A").format.columnWidth = 22;
compareSheet.getRange(`B:${compareLastCol}`).format.columnWidth = 19;
compareSheet.getRange(`A4:${compareLastCol}7`).format.font.name = FONT;
compareSheet.getRange("D5:D7").format.numberFormat = "0.00%";
const differenceStart = 9;
compareSheet.getRange(`A${differenceStart}`).values = [["LP 与 MILP 等价最优解差异"]];
compareSheet.getRange(`A${differenceStart}`).format = { font: { name: FONT, size: 11, bold: true } };
compareSheet.getRange(`A${differenceStart + 1}`).write([compareData.differenceColumns, ...compareData.differenceRows]);
styleHeader(compareSheet, `A${differenceStart + 1}:C${differenceStart + 1}`);
compareSheet.getRange(`B${differenceStart + 2}:B${differenceStart + 5}`).format.numberFormat = "#,##0.000000";
compareSheet.getRange("A:A").format.columnWidth = 25;
compareSheet.getRange("B:C").format.columnWidth = 22;
records.push(await exportAndVerify(compareWorkbook, path.join(outputDir, "问题一_三种方案结果对比.xlsx"), "三种方案对比"));

await fs.writeFile(path.join(root, "tmp", "问题一_结果工作簿检查.json"), `${JSON.stringify(records, null, 2)}\n`, "utf8");
console.log(JSON.stringify(records.map(({ outputPath }) => outputPath), null, 2));

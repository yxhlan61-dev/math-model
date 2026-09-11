import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve(import.meta.dirname, "..");
const outputDir = path.join(root, "code", "outputs", "问题二");
const intermediateDir = path.join(root, "tmp", "问题二_求解中间结果");
const previewDir = path.join(root, "tmp", "问题二_结果工作簿预览");
const templatePath = path.join(root, "附件", "附件5", "result2.xlsx");
const bundlePath = path.join(intermediateDir, "问题二_工作簿数据.json");
const artifactToolEntry = process.env.CODEX_ARTIFACT_TOOL_PATH ?? path.join(
  os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies",
  "node", "node_modules", "@oai", "artifact-tool", "dist", "artifact_tool.mjs",
);
const { FileBlob, SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactToolEntry).href);

const FONT = "Arial";
const HEADER = "#1F4E78";
const LIGHT_BLUE = "#D9EAF7";
const segments = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"];

function dateValue(text) {
  return new Date(`${text}T00:00:00+08:00`);
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

function excelColumnName(indexOneBased) {
  let value = indexOneBased;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

async function renderRange(workbook, sheetName, range, fileName) {
  const preview = await workbook.render({ sheetName, range, scale: 1.4, format: "png" });
  await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}

async function exportWorkbook(workbook, outputPath) {
  workbook.recalculate();
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "最终公式错误扫描",
  });
  const blob = await SpreadsheetFile.exportXlsx(workbook);
  await blob.save(outputPath);
  return errors.ndjson;
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const bundle = JSON.parse(await fs.readFile(bundlePath, "utf8"));

// 一、按附件5模板生成提交工作簿。
const submission = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
const purchaseSheet = submission.worksheets.getItem("计划购电量");
const storageSheet = submission.worksheets.getItem("充放电量");
const emergencySheet = submission.worksheets.getItem("紧急购电量");

const purchaseRows = bundle.purchaseRows.map((row) => [
  dateValue(row.date), ...row.purchase, row.total_purchase, row.total_cost,
]);
purchaseSheet.getRange("A2").write(purchaseRows);
purchaseSheet.getRange("A2:A335").setNumberFormat("yyyy-mm-dd");
purchaseSheet.getRange("B2:EQ335").format.numberFormat = "#,##0.000000";
purchaseSheet.freezePanes.freezeRows(1);
purchaseSheet.freezePanes.freezeColumns(1);

const storageRows = [];
for (const row of bundle.storageRows) {
  for (let i = 0; i < 6; i += 1) {
    storageRows.push([
      i === 0 ? dateValue(row.date) : null,
      segments[i],
      row.charge_4h[i],
      row.discharge_4h[i],
      i === 0 ? "0:00" : (i === 1 ? "24:00" : null),
      i === 0 ? row.initial_soc : (i === 1 ? row.terminal_soc : null),
    ]);
  }
}
storageSheet.getRange("A2:F5000").clear({ applyTo: "contents" });
storageSheet.getRange("A2").write(storageRows);
const storageLastRow = storageRows.length + 1;
storageSheet.getRange(`A2:A${storageLastRow}`).setNumberFormat("yyyy-mm-dd");
storageSheet.getRange(`C2:D${storageLastRow}`).format.numberFormat = "#,##0.000000";
storageSheet.getRange(`F2:F${storageLastRow}`).format.numberFormat = "#,##0.000000";
storageSheet.getRange(`A1:F${storageLastRow}`).format.font = { name: FONT, size: 9, color: "#222222" };
styleHeader(storageSheet, "A1:F1");
storageSheet.getRange("A:A").format.columnWidth = 13;
storageSheet.getRange("B:B").format.columnWidth = 16;
storageSheet.getRange("C:D").format.columnWidth = 15;
storageSheet.getRange("E:E").format.columnWidth = 11;
storageSheet.getRange("F:F").format.columnWidth = 16;
storageSheet.freezePanes.freezeRows(1);

const emergencyRows = [];
let priorDate = null;
for (const row of bundle.emergencyRows) {
  emergencyRows.push([
    row["日期"] !== priorDate ? dateValue(row["日期"]) : null,
    row["紧急购电时间段"],
    row["紧急购电量(kWh)"],
  ]);
  priorDate = row["日期"];
}
emergencySheet.getRange("A2:C20000").clear({ applyTo: "contents" });
if (emergencyRows.length > 0) emergencySheet.getRange("A2").write(emergencyRows);
const emergencyLastRow = Math.max(2, emergencyRows.length + 1);
emergencySheet.getRange(`A2:A${emergencyLastRow}`).setNumberFormat("yyyy-mm-dd");
emergencySheet.getRange(`C2:C${emergencyLastRow}`).format.numberFormat = "#,##0.000000";
emergencySheet.getRange(`A1:C${emergencyLastRow}`).format.font = { name: FONT, size: 9, color: "#222222" };
styleHeader(emergencySheet, "A1:C1");
emergencySheet.getRange("A:A").format.columnWidth = 14;
emergencySheet.getRange("B:B").format.columnWidth = 24;
emergencySheet.getRange("C:C").format.columnWidth = 18;
emergencySheet.freezePanes.freezeRows(1);

submission.recalculate();
const submissionInspect = await submission.inspect({
  kind: "sheet,region",
  maxChars: 10000,
  tableMaxRows: 8,
  tableMaxCols: 8,
});
await renderRange(submission, "计划购电量", "A1:L8", "提交_计划购电量_首部.png");
await renderRange(submission, "计划购电量", "EN1:EQ8", "提交_计划购电量_合计.png");
await renderRange(submission, "充放电量", "A1:F20", "提交_充放电量_首部.png");
await renderRange(submission, "紧急购电量", `A1:C${Math.min(30, emergencyLastRow)}`, "提交_紧急购电量_首部.png");
const submissionPath = path.join(outputDir, "问题二_MILP_求解结果_result2提交格式.xlsx");
const submissionErrors = await exportWorkbook(submission, submissionPath);

// 二、生成便于复核与论文取数的审计工作簿。
const audit = Workbook.create();
const summarySheet = audit.worksheets.add("结果摘要");
const comparisonSheet = audit.worksheets.add("方案对比");
const dailySheet = audit.worksheets.add("逐日汇总");
summarySheet.showGridLines = false;
comparisonSheet.showGridLines = false;
dailySheet.showGridLines = false;
summarySheet.tabColor = HEADER;
comparisonSheet.tabColor = "#70AD47";
dailySheet.tabColor = "#5B9BD5";

summarySheet.getRange("A2").values = [["问题二周期预测与日备不足惩罚MILP结果"]];
summarySheet.getRange("A2:D2").format.font = { name: FONT, size: 14, bold: true, color: "#222222" };
summarySheet.getRange("A4:B4").values = [["指标", "结果"]];
styleHeader(summarySheet, "A4:B4");
const summaryRows = [
  ["正式区间", bundle.summary["正式区间"]],
  ["正式天数", bundle.summary["正式天数"]],
  ["计划购电量(kWh)", bundle.summary["计划购电量_kWh"]],
  ["实际紧急购电量(kWh)", bundle.summary["实际紧急购电量_kWh"]],
  ["实际富余电量(kWh)", bundle.summary["实际富余电量_kWh"]],
  ["充电量(kWh)", bundle.summary["充电量_kWh"]],
  ["放电量(kWh)", bundle.summary["放电量_kWh"]],
  ["正常购电费(元)", bundle.summary["正常购电费_元"]],
  ["紧急购电费(元)", bundle.summary["紧急购电费_元"]],
  ["实际总费用(元)", bundle.summary["实际总费用_元"]],
  ["日备目标(kWh)", bundle.summary["目标函数参数"]["日备目标_kWh"]],
  ["日备不足惩罚系数(元/kWh)", bundle.summary["目标函数参数"]["日备不足惩罚系数_元每kWh"]],
  ["日备不足量(kWh)", bundle.summary["日备不足量_kWh"]],
  ["日备不足惩罚(元)", bundle.summary["日备不足惩罚_元"]],
  ["平均日末储电量(kWh)", bundle.summary["平均日末储电量_kWh"]],
  ["紧急购电时段数", bundle.summary["发生紧急购电的时段数"]],
  ["2月1日初始储电量(kWh)", bundle.summary["2月1日初始储电量_kWh"]],
  ["12月31日末储电量(kWh)", bundle.summary["12月31日末储电量_kWh"]],
  ["负荷MAE(kWh)", bundle.summary["预测指标"]["负荷"]["MAE(kWh)"]],
  ["负荷RMSE(kWh)", bundle.summary["预测指标"]["负荷"]["RMSE(kWh)"]],
  ["负荷WAPE", bundle.summary["预测指标"]["负荷"]["WAPE"]],
  ["光伏MAE(kWh)", bundle.summary["预测指标"]["光伏_全时段"]["MAE(kWh)"]],
  ["光伏RMSE(kWh)", bundle.summary["预测指标"]["光伏_全时段"]["RMSE(kWh)"]],
  ["光伏WAPE", bundle.summary["预测指标"]["光伏_全时段"]["WAPE"]],
  ["最大场景平衡残差(kWh)", bundle.summary["约束校验"]["max_abs_scenario_balance_residual_kwh"]],
  ["最大实际平衡残差(kWh)", bundle.summary["约束校验"]["max_abs_actual_balance_residual_kwh"]],
  ["最大SOC递推残差(kWh)", bundle.summary["约束校验"]["max_abs_soc_recursion_residual_kwh"]],
  ["同时充放电时段数", bundle.summary["约束校验"]["simultaneous_charge_discharge_count"]],
  ["最大MIP相对间隙", bundle.summary["约束校验"]["max_mip_gap"]],
];
summarySheet.getRange("A5").write(summaryRows);
summarySheet.getRange(`B5:B${summaryRows.length + 4}`).format.numberFormat = "#,##0.000000";
summarySheet.getRange("B25:B25").format.numberFormat = "0.00%";
summarySheet.getRange("B28:B28").format.numberFormat = "0.00%";
summarySheet.getRange("A:A").format.columnWidth = 34;
summarySheet.getRange("B:B").format.columnWidth = 24;
summarySheet.getRange(`A4:B${summaryRows.length + 4}`).format.font.name = FONT;

comparisonSheet.getRange("A2").values = [["有无日备不足惩罚方案对比"]];
comparisonSheet.getRange("A2:D2").format.font = { name: FONT, size: 14, bold: true, color: "#222222" };
comparisonSheet.getRange("A4:D4").values = [["指标", "无日备不足惩罚", "有日备不足惩罚", "差值（有-无）"]];
styleHeader(comparisonSheet, "A4:D4");
const comparisonMetrics = [
  ["计划购电量(kWh)", "计划购电量_kWh"],
  ["实际紧急购电量(kWh)", "实际紧急购电量_kWh"],
  ["充电量(kWh)", "充电量_kWh"],
  ["放电量(kWh)", "放电量_kWh"],
  ["正常购电费(元)", "正常购电费_元"],
  ["紧急购电费(元)", "紧急购电费_元"],
  ["实际购电总费用(元)", "实际购电总费用_元"],
  ["平均日末储电量(kWh)", "平均日末储电量_kWh"],
  ["日末储电量标准差(kWh)", "日末储电量标准差_kWh"],
  ["日末低于6000kWh天数", "日末低于6000kWh天数"],
  ["12月31日末储电量(kWh)", "12月31日末储电量_kWh"],
];
const comparisonRows = comparisonMetrics.map(([label, key]) => [
  label,
  bundle.comparison["无日备不足惩罚"][key],
  bundle.comparison["有日备不足惩罚"][key],
  bundle.comparison["差值_有减无"][key],
]);
comparisonSheet.getRange("A5").write(comparisonRows);
comparisonSheet.getRange(`B5:D${comparisonRows.length + 4}`).format.numberFormat = "#,##0.000000";
comparisonSheet.getRange(`A4:D${comparisonRows.length + 4}`).format.font = { name: FONT, size: 9, color: "#222222" };
styleHeader(comparisonSheet, "A4:D4");
comparisonSheet.getRange("A:A").format.columnWidth = 32;
comparisonSheet.getRange("B:D").format.columnWidth = 22;
comparisonSheet.freezePanes.freezeRows(4);

const emergencyByDate = new Map();
for (const row of bundle.emergencyRows) {
  emergencyByDate.set(row["日期"], (emergencyByDate.get(row["日期"]) ?? 0) + row["紧急购电量(kWh)"]);
}
const dailyRows = bundle.purchaseRows.map((purchase, index) => {
  const storage = bundle.storageRows[index];
  return [
    dateValue(purchase.date),
    purchase.total_purchase,
    emergencyByDate.get(purchase.date) ?? 0,
    storage.charge_4h.reduce((a, b) => a + b, 0),
    storage.discharge_4h.reduce((a, b) => a + b, 0),
    storage.initial_soc,
    storage.terminal_soc,
    purchase.total_cost,
  ];
});
dailySheet.getRange("A2").values = [["日期", "计划购电量(kWh)", "紧急购电量(kWh)", "充电量(kWh)", "放电量(kWh)", "日初储电量(kWh)", "日末储电量(kWh)", "总费用(元)"]];
styleHeader(dailySheet, "A2:H2");
dailySheet.getRange("A3").write(dailyRows);
const dailyLastRow = dailyRows.length + 2;
dailySheet.getRange(`A3:A${dailyLastRow}`).setNumberFormat("yyyy-mm-dd");
dailySheet.getRange(`B3:H${dailyLastRow}`).format.numberFormat = "#,##0.000000";
dailySheet.getRange(`A2:H${dailyLastRow}`).format.font = { name: FONT, size: 9, color: "#222222" };
styleHeader(dailySheet, "A2:H2");
dailySheet.getRange("A:A").format.columnWidth = 14;
dailySheet.getRange("B:H").format.columnWidth = 19;
dailySheet.freezePanes.freezeRows(2);
dailySheet.freezePanes.freezeColumns(1);

audit.recalculate();
await renderRange(audit, "结果摘要", `A1:B${summaryRows.length + 6}`, "审计_结果摘要.png");
await renderRange(audit, "方案对比", `A1:D${comparisonRows.length + 6}`, "审计_方案对比.png");
await renderRange(audit, "逐日汇总", "A1:H18", "审计_逐日汇总.png");
const auditPath = path.join(outputDir, "问题二_MILP_全年结果与校验.xlsx");
const auditErrors = await exportWorkbook(audit, auditPath);

const records = {
  outputs: [submissionPath, auditPath],
  submissionInspect: submissionInspect.ndjson,
  submissionFormulaErrors: submissionErrors,
  auditFormulaErrors: auditErrors,
  dimensions: {
    purchase: `A1:EQ${purchaseRows.length + 1}`,
    storage: `A1:F${storageLastRow}`,
    emergency: `A1:C${emergencyLastRow}`,
  },
};
await fs.writeFile(path.join(intermediateDir, "问题二_结果工作簿检查.json"), `${JSON.stringify(records, null, 2)}\n`, "utf8");
console.log(JSON.stringify(records, null, 2));

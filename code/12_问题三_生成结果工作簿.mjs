import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve(import.meta.dirname, "..");
const outputDir = path.join(root, "code", "outputs", "问题三");
const intermediateDir = path.join(root, "tmp", "问题三_求解中间结果");
const previewDir = path.join(root, "tmp", "问题三_结果工作簿预览");
const templatePath = path.join(root, "附件", "附件5", "result3.xlsx");
const bundlePath = path.join(intermediateDir, "问题三_工作簿数据.json");
const artifactToolEntry = process.env.CODEX_ARTIFACT_TOOL_PATH ?? path.join(
  os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies",
  "node", "node_modules", "@oai", "artifact-tool", "dist", "artifact_tool.mjs",
);
const { FileBlob, SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactToolEntry).href);

const FONT = "Arial";
const HEADER = "#1F4E78";
const GREEN = "#548235";
const segments = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"];

function dateValue(text) { return new Date(`${text}T00:00:00+08:00`); }
function styleHeader(sheet, range, fill = HEADER) {
  sheet.getRange(range).format = {
    fill,
    font: { name: FONT, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
  };
}
async function render(workbook, sheetName, range, fileName) {
  const image = await workbook.render({ sheetName, range, scale: 1.4, format: "png" });
  await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await image.arrayBuffer()));
}
async function exportChecked(workbook, outputPath) {
  workbook.recalculate();
  const errors = await workbook.inspect({
    kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 }, summary: "最终公式错误扫描",
  });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  return errors.ndjson;
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const bundle = JSON.parse(await fs.readFile(bundlePath, "utf8"));

// 一、按附件5原结构填写正式提交文件。
const submission = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
const planSheet = submission.worksheets.getItem("计划购电量");
const adjustmentSheet = submission.worksheets.getItem("调整购电量");
const storageSheet = submission.worksheets.getItem("充放电量");
const emergencySheet = submission.worksheets.getItem("紧急购电量");

const planRows = bundle.purchaseRows.map((row) => [dateValue(row.date), ...row.values, row.total, row.cost]);
const adjustmentRows = bundle.adjustmentRows.map((row) => [dateValue(row.date), ...row.values, row.total, row.cost]);
planSheet.getRange("A2").write(planRows);
adjustmentSheet.getRange("A2").write(adjustmentRows);
for (const sheet of [planSheet, adjustmentSheet]) {
  sheet.getRange("A2:A335").setNumberFormat("yyyy-mm-dd");
  sheet.getRange("B2:EQ335").format.numberFormat = "#,##0.000000";
  sheet.freezePanes.freezeRows(1); sheet.freezePanes.freezeColumns(1);
}

const storageRows = [];
for (const row of bundle.storageRows) {
  for (let i = 0; i < 6; i += 1) {
    storageRows.push([
      i === 0 ? dateValue(row.date) : null, segments[i], row.charge_4h[i], row.discharge_4h[i],
      i === 0 ? "0:00" : (i === 1 ? "24:00" : null),
      i === 0 ? row.initial_soc : (i === 1 ? row.terminal_soc : null),
    ]);
  }
}
storageSheet.getRange("A2:F5000").clear({ applyTo: "contents" });
storageSheet.getRange("A2").write(storageRows);
const storageEnd = storageRows.length + 1;
storageSheet.getRange(`A2:A${storageEnd}`).setNumberFormat("yyyy-mm-dd");
storageSheet.getRange(`C2:D${storageEnd}`).format.numberFormat = "#,##0.000000";
storageSheet.getRange(`F2:F${storageEnd}`).format.numberFormat = "#,##0.000000";
storageSheet.getRange(`A2:F${storageEnd}`).format.font = { name: FONT, size: 9, color: "#222222" };

const emergencyRows = [];
let priorDate = null;
for (const row of bundle.emergencyRows) {
  emergencyRows.push([row.date !== priorDate ? dateValue(row.date) : null, row.period, row.amount]);
  priorDate = row.date;
}
emergencySheet.getRange("A2:C20000").clear({ applyTo: "contents" });
if (emergencyRows.length) emergencySheet.getRange("A2").write(emergencyRows);
const emergencyEnd = Math.max(2, emergencyRows.length + 1);
emergencySheet.getRange(`A2:A${emergencyEnd}`).setNumberFormat("yyyy-mm-dd");
emergencySheet.getRange(`C2:C${emergencyEnd}`).format.numberFormat = "#,##0.000000";
emergencySheet.getRange(`A2:C${emergencyEnd}`).format.font = { name: FONT, size: 9, color: "#222222" };

await render(submission, "计划购电量", "A1:L8", "提交_计划购电量.png");
await render(submission, "调整购电量", "A1:L8", "提交_调整购电量.png");
await render(submission, "计划购电量", "EN1:EQ8", "提交_计划合计.png");
await render(submission, "调整购电量", "EN1:EQ8", "提交_调整合计.png");
await render(submission, "充放电量", "A1:F20", "提交_充放电量.png");
await render(submission, "紧急购电量", `A1:C${Math.min(30, emergencyEnd)}`, "提交_紧急购电量.png");
const submissionPath = path.join(outputDir, "问题三_四时点滚动MILP_result3提交格式.xlsx");
const submissionErrors = await exportChecked(submission, submissionPath);

// 二、生成复核与论文取数用审计工作簿。
const audit = Workbook.create();
const summary = audit.worksheets.add("结果摘要");
const comparison = audit.worksheets.add("策略对比");
const daily = audit.worksheets.add("逐日汇总");
for (const sheet of [summary, comparison, daily]) sheet.showGridLines = false;
summary.tabColor = HEADER; comparison.tabColor = GREEN; daily.tabColor = "#5B9BD5";

summary.getRange("A2").values = [["问题三官方光伏预报与四时点滚动MILP结果"]];
summary.getRange("A2:D2").format.font = { name: FONT, size: 14, bold: true, color: "#222222" };
summary.getRange("A4:B4").values = [["指标", "结果"]]; styleHeader(summary, "A4:B4");
const official = bundle.policySummaries["S61218_全部四时点"];
const summaryRows = [
  ["正式区间", "2025-02-01至2025-12-31"], ["正式天数", official["正式天数"]],
  ["计划购电量(kWh)", official["计划购电量(kWh)"]], ["调整后购电量(kWh)", official["调整后购电量(kWh)"]],
  ["上调量(kWh)", official["上调量(kWh)"]], ["下调量(kWh)", official["下调量(kWh)"]],
  ["紧急购电量(kWh)", official["紧急购电量(kWh)"]], ["富余电量(kWh)", official["富余电量(kWh)"]],
  ["计划购电费(元)", official["计划购电费(元)"]], ["上调费用(元)", official["上调费用(元)"]],
  ["下调抵扣(元)", official["下调抵扣(元)"]], ["计划与调整结算费(元)", official["计划与调整结算费(元)"]],
  ["紧急购电费(元)", official["紧急购电费(元)"]], ["实际总费用(元)", official["实际总费用(元)"]],
  ["平均日末储电量(kWh)", official["平均日末储电量(kWh)"]],
  ["6点发生调整天数", official["6点发生调整天数"]], ["12点发生调整天数", official["12点发生调整天数"]],
  ["18点发生调整天数", official["18点发生调整天数"]], ["12月31日末储电量(kWh)", official["12月31日末储电量(kWh)"]],
  ["最大场景平衡残差(kWh)", bundle.validations["S61218_全部四时点"]["max_abs_scenario_balance_residual_kwh"]],
  ["最大SOC递推残差(kWh)", bundle.validations["S61218_全部四时点"]["max_abs_soc_recursion_residual_kwh"]],
  ["最大MIP相对间隙", bundle.validations["S61218_全部四时点"]["max_mip_gap"]],
];
summary.getRange("A5").write(summaryRows);
summary.getRange(`B5:B${summaryRows.length + 4}`).format.numberFormat = "#,##0.000000";
summary.getRange(`A4:B${summaryRows.length + 4}`).format.font = { name: FONT, size: 10, color: "#222222" };
styleHeader(summary, "A4:B4"); summary.getRange("A:A").format.columnWidth = 34; summary.getRange("B:B").format.columnWidth = 26;

comparison.getRange("A2").values = [["四种日内预报更新策略对比"]];
comparison.getRange("A2:F2").format.font = { name: FONT, size: 14, bold: true, color: "#222222" };
comparison.getRange("A4:F4").values = [["策略", "计划与调整结算费(元)", "紧急购电费(元)", "实际总费用(元)", "紧急购电量(kWh)", "平均日末储电量(kWh)"]];
styleHeader(comparison, "A4:F4");
const policyOrder = ["S0_仅0点", "S6_增加6点", "S612_增加6点12点", "S61218_全部四时点"];
const comparisonRows = policyOrder.map((name) => {
  const value = bundle.policySummaries[name];
  return [name, value["计划与调整结算费(元)"], value["紧急购电费(元)"], value["实际总费用(元)"], value["紧急购电量(kWh)"], value["平均日末储电量(kWh)"]];
});
comparison.getRange("A5").write(comparisonRows);
comparison.getRange("B5:F8").format.numberFormat = "#,##0.000000";
comparison.getRange("A10:D10").values = [["新增预报时点", "边际费用变化(元)", "边际紧急购电量变化(kWh)", "判断"]]; styleHeader(comparison, "A10:D10", GREEN);
const marginalRows = [];
for (let i = 1; i < policyOrder.length; i += 1) {
  const prior = bundle.policySummaries[policyOrder[i - 1]]; const current = bundle.policySummaries[policyOrder[i]];
  marginalRows.push([["6:00", "12:00", "18:00"][i - 1], current["实际总费用(元)"] - prior["实际总费用(元)"],
    current["紧急购电量(kWh)"] - prior["紧急购电量(kWh)"], i < 3 ? "价值显著" : "价值较小但为正"]);
}
comparison.getRange("A11").write(marginalRows); comparison.getRange("B11:C13").format.numberFormat = "#,##0.000000";
comparison.getRange("A:A").format.columnWidth = 24; comparison.getRange("B:F").format.columnWidth = 23;
comparison.freezePanes.freezeRows(4);

const dailyHeaders = ["日期", "计划购电量(kWh)", "调整后购电量(kWh)", "上调量(kWh)", "下调量(kWh)", "紧急购电量(kWh)", "计划购电费(元)", "上调费用(元)", "下调抵扣(元)", "计划与调整结算费(元)", "紧急购电费(元)", "实际总费用(元)", "日初储电量(kWh)", "日末储电量(kWh)"];
daily.getRange("A2:N2").values = [dailyHeaders]; styleHeader(daily, "A2:N2");
const dailyRows = bundle.dailyRows.map((row) => [dateValue(row["日期"]), ...dailyHeaders.slice(1).map((header) => row[header])]);
daily.getRange("A3").write(dailyRows); const dailyEnd = dailyRows.length + 2;
daily.getRange(`A3:A${dailyEnd}`).setNumberFormat("yyyy-mm-dd"); daily.getRange(`B3:N${dailyEnd}`).format.numberFormat = "#,##0.000000";
daily.getRange(`A2:N${dailyEnd}`).format.font = { name: FONT, size: 9, color: "#222222" }; styleHeader(daily, "A2:N2");
daily.getRange("A:A").format.columnWidth = 14; daily.getRange("B:N").format.columnWidth = 19;
daily.freezePanes.freezeRows(2); daily.freezePanes.freezeColumns(1);

await render(audit, "结果摘要", `A1:B${summaryRows.length + 6}`, "审计_结果摘要.png");
await render(audit, "策略对比", "A1:F15", "审计_策略对比.png");
await render(audit, "逐日汇总", "A1:N16", "审计_逐日汇总.png");
const auditPath = path.join(outputDir, "问题三_四时点滚动MILP_全年结果与校验.xlsx");
const auditErrors = await exportChecked(audit, auditPath);

const result = {
  outputs: [submissionPath, auditPath], submissionFormulaErrors: submissionErrors, auditFormulaErrors: auditErrors,
  dimensions: { plan: `A1:EQ${planRows.length + 1}`, adjustment: `A1:EQ${adjustmentRows.length + 1}`,
    storage: `A1:F${storageEnd}`, emergency: `A1:C${emergencyEnd}` }, previewDir,
};
await fs.writeFile(path.join(intermediateDir, "问题三_结果工作簿检查.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
console.log(JSON.stringify(result, null, 2));

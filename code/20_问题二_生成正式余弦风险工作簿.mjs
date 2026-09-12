import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve(import.meta.dirname, "..");
const intermediateDir = path.join(root, "tmp", "问题二_求解中间结果");
const outputDir = path.join(root, "code", "outputs", "问题二");
const templatePath = path.join(root, "附件", "附件5", "result2.xlsx");
const bundle = JSON.parse(await fs.readFile(path.join(intermediateDir, "问题二_余弦风险_工作簿数据.json"), "utf8"));
const artifactToolEntry = process.env.CODEX_ARTIFACT_TOOL_PATH ?? path.join(
  os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies",
  "node", "node_modules", "@oai", "artifact-tool", "dist", "artifact_tool.mjs",
);
const { FileBlob, SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactToolEntry).href);

const FONT = "Arial";
const HEADER = "#1F4E78";
const segments = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"];
const dateValue = (text) => new Date(`${text}T00:00:00+08:00`);

function header(sheet, range) {
  sheet.getRange(range).format = {
    fill: HEADER,
    font: { name: FONT, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
  };
}
async function exportWorkbook(workbook, outputPath) {
  workbook.recalculate();
  const errors = await workbook.inspect({
    kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 }, summary: "公式错误扫描",
  });
  const blob = await SpreadsheetFile.exportXlsx(workbook);
  await blob.save(outputPath);
  return errors.ndjson;
}

await fs.mkdir(outputDir, { recursive: true });
const submission = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
const purchase = submission.worksheets.getItem("计划购电量");
const storage = submission.worksheets.getItem("充放电量");
const emergency = submission.worksheets.getItem("紧急购电量");
const purchaseRows = bundle.purchaseRows.map((r) => [dateValue(r.date), ...r.purchase, r.total_purchase, r.total_cost]);
purchase.getRange("A2").write(purchaseRows);
purchase.getRange("A2:A335").setNumberFormat("yyyy-mm-dd");
purchase.getRange("B2:EQ335").format.numberFormat = "#,##0.000000";

const storageRows = [];
for (const r of bundle.storageRows) for (let i = 0; i < 6; i += 1) storageRows.push([
  i === 0 ? dateValue(r.date) : null, segments[i], r.charge_4h[i], r.discharge_4h[i],
  i === 0 ? "0:00" : (i === 1 ? "24:00" : null), i === 0 ? r.initial_soc : (i === 1 ? r.terminal_soc : null),
]);
storage.getRange("A2:F5000").clear({ applyTo: "contents" });
storage.getRange("A2").write(storageRows);
storage.getRange(`A2:A${storageRows.length + 1}`).setNumberFormat("yyyy-mm-dd");
storage.getRange(`C2:F${storageRows.length + 1}`).format.numberFormat = "#,##0.000000";

const emergencyRows = [];
let prior = null;
for (const r of bundle.emergencyRows) {
  emergencyRows.push([r.date === prior ? null : dateValue(r.date), r.interval, r.emergency_kwh]);
  prior = r.date;
}
emergency.getRange("A2:C20000").clear({ applyTo: "contents" });
emergency.getRange("A2").write(emergencyRows);
emergency.getRange(`A2:A${Math.max(2, emergencyRows.length + 1)}`).setNumberFormat("yyyy-mm-dd");
emergency.getRange(`C2:C${Math.max(2, emergencyRows.length + 1)}`).format.numberFormat = "#,##0.000000";
const submissionPath = path.join(outputDir, "问题二_MILP_求解结果_result2提交格式.xlsx");
const submissionErrors = await exportWorkbook(submission, submissionPath);

const audit = Workbook.create();
const summary = audit.worksheets.add("结果摘要");
const compare = audit.worksheets.add("方案对比");
const daily = audit.worksheets.add("逐日汇总");
for (const sheet of [summary, compare, daily]) sheet.showGridLines = false;
summary.getRange("A2:B2").merge(); summary.getRange("A2").values = [["问题二：季节SOC与因果场景风险微调MILP"]];
summary.getRange("A2").format.font = { name: FONT, size: 14, bold: true, color: "#222222" };
summary.getRange("A4:B4").values = [["指标", "结果"]]; header(summary, "A4:B4");
const s = bundle.summary;
const summaryRows = [
  ["正式区间", s.official_interval], ["正式天数", s.days], ["正常购电费(元)", s.normal_cost_yuan],
  ["紧急购电费(元)", s.emergency_cost_yuan], ["实际总费用(元)", s.total_cost_yuan],
  ["计划购电量(kWh)", s.purchase_kwh], ["紧急购电量(kWh)", s.emergency_kwh], ["实际富余量(kWh)", s.surplus_kwh],
  ["平均日末储电量(kWh)", s.end_soc_mean_kwh], ["日末储电量标准差(kWh)", s.end_soc_std_kwh],
  ["12月31日末SOC(kWh)", s.end_soc_last_kwh], ["风险微调均值(kWh)", s.risk_adjust_mean_kwh],
  ["风险微调标准差(kWh)", s.risk_adjust_std_kwh], ["风险上调天数", s.risk_positive_days],
  ["风险下调天数", s.risk_negative_days], ["达到+600kWh上限天数", s.risk_cap_days],
  ["贴微调后下沿天数", s.at_lower_band_days],
];
summary.getRange("A5").write(summaryRows); summary.getRange(`B5:B${summaryRows.length + 4}`).format.numberFormat = "#,##0.000000";
summary.getRange("A:A").format.columnWidth = 34; summary.getRange("B:B").format.columnWidth = 24;

compare.getRange("A2:D2").merge(); compare.getRange("A2").values = [["新方案与无SOC终端约束方案对比"]];
compare.getRange("A2").format.font = { name: FONT, size: 14, bold: true, color: "#222222" };
compare.getRange("A4:D4").values = [["指标", "无SOC终端约束", "季节SOC+风险微调", "差值（新-无约束）"]]; header(compare, "A4:D4");
const compareRows = [
  ["实际总费用(元)", s.no_terminal_soc_total_yuan, s.total_cost_yuan, s.difference_new_minus_no_terminal_soc_yuan],
  ["平均日末SOC(kWh)", s.no_terminal_soc_end_mean_kwh, s.end_soc_mean_kwh, s.end_soc_mean_kwh-s.no_terminal_soc_end_mean_kwh],
  ["12月31日末SOC(kWh)", s.no_terminal_soc_end_last_kwh, s.end_soc_last_kwh, s.end_soc_last_kwh-s.no_terminal_soc_end_last_kwh],
];
compare.getRange("A5").write(compareRows); compare.getRange("B5:D7").format.numberFormat = "#,##0.000000";
compare.getRange("A:A").format.columnWidth = 30; compare.getRange("B:D").format.columnWidth = 22;

daily.getRange("A2:K2").values = [["日期", "正常购电量(kWh)", "紧急购电量(kWh)", "充电量(kWh)", "放电量(kWh)", "日初SOC(kWh)", "日末SOC(kWh)", "季节基准SOC(kWh)", "风险微调(kWh)", "实际总费用(元)", "风险分数(kWh)"]]; header(daily, "A2:K2");
const dailyRows = bundle.dailyRows.map((r, index) => [dateValue(r.date), bundle.purchaseRows[index].total_purchase, r.emergency_kwh, r.charge_kwh, r.discharge_kwh, r.initial_soc_kwh, r.end_soc_kwh, r.seasonal_reference_soc_kwh, r.daily_adjustment_kwh, r.total_cost_yuan, r.risk_score_kwh]);
daily.getRange("A3").write(dailyRows); daily.getRange(`A3:A${dailyRows.length + 2}`).setNumberFormat("yyyy-mm-dd"); daily.getRange(`B3:K${dailyRows.length + 2}`).format.numberFormat = "#,##0.000000";
daily.getRange("A:A").format.columnWidth = 14; daily.getRange("B:K").format.columnWidth = 18;
const auditPath = path.join(outputDir, "问题二_MILP_全年结果与校验.xlsx");
const auditErrors = await exportWorkbook(audit, auditPath);
const records = { outputs: [submissionPath, auditPath], submissionFormulaErrors: submissionErrors, auditFormulaErrors: auditErrors, dimensions: { purchase: "A1:EQ335", storage: `A1:F${storageRows.length + 1}`, emergency: `A1:C${emergencyRows.length + 1}` } };
await fs.writeFile(path.join(intermediateDir, "问题二_余弦风险_工作簿检查.json"), `${JSON.stringify(records, null, 2)}\n`, "utf8");
console.log(JSON.stringify(records, null, 2));

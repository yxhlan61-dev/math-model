import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = path.resolve(import.meta.dirname, "..");
const outputDir = path.join(root, "code", "outputs", "问题四");
const intermediateDir = path.join(root, "tmp", "问题四_求解中间结果");
const previewDir = path.join(root, "tmp", "问题四_结果工作簿预览");
const artifactToolEntry = process.env.CODEX_ARTIFACT_TOOL_PATH ?? path.join(
  os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies",
  "node", "node_modules", "@oai", "artifact-tool", "dist", "artifact_tool.mjs",
);
const { FileBlob, SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactToolEntry).href);

const bundle = JSON.parse(await fs.readFile(path.join(intermediateDir, "问题四_工作簿数据.json"), "utf8"));
const FONT = "Arial";
const HEADER = "#1F4E78";
const GREEN = "#548235";
const segments = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"];

function dateValue(text) { return new Date(`${text}T00:00:00+08:00`); }
function header(sheet, range, fill = HEADER) {
  sheet.getRange(range).format = {
    fill, font: { name: FONT, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
  };
}
async function render(workbook, sheetName, range, fileName) {
  const image = await workbook.render({ sheetName, range, scale: 1.3, format: "png" });
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

async function makeSubmission(templateName, data, adjusted) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(
    path.join(root, "附件", "附件5", templateName),
  ));
  const plan = workbook.worksheets.getItem("计划购电量");
  const planRows = data.purchaseRows.map((row) => [dateValue(row.date), ...row.values, row.total, row.cost]);
  plan.getRange("A2").write(planRows);
  plan.getRange("A2:A335").setNumberFormat("yyyy-mm-dd");
  plan.getRange("B2:EQ335").format.numberFormat = "#,##0.000000";
  plan.freezePanes.freezeRows(1); plan.freezePanes.freezeColumns(1);

  if (adjusted) {
    const adjustment = workbook.worksheets.getItem("调整购电量");
    const rows = data.adjustmentRows.map((row) => [dateValue(row.date), ...row.values, row.total, row.cost]);
    adjustment.getRange("A2").write(rows);
    adjustment.getRange("A2:A335").setNumberFormat("yyyy-mm-dd");
    adjustment.getRange("B2:EQ335").format.numberFormat = "#,##0.000000";
    adjustment.freezePanes.freezeRows(1); adjustment.freezePanes.freezeColumns(1);
  }

  const storage = workbook.worksheets.getItem("充放电量");
  const storageRows = [];
  for (const row of data.storageRows) {
    for (let i = 0; i < 6; i += 1) {
      storageRows.push([
        i === 0 ? dateValue(row.date) : null, segments[i], row.charge_4h[i], row.discharge_4h[i],
        i === 0 ? "0:00" : (i === 1 ? "24:00" : null),
        i === 0 ? row.initial_soc : (i === 1 ? row.terminal_soc : null),
      ]);
    }
  }
  storage.getRange("A2:F5000").clear({ applyTo: "contents" });
  storage.getRange("A2").write(storageRows);
  const storageEnd = storageRows.length + 1;
  storage.getRange(`A2:A${storageEnd}`).setNumberFormat("yyyy-mm-dd");
  storage.getRange(`C2:D${storageEnd}`).format.numberFormat = "#,##0.000000";
  storage.getRange(`F2:F${storageEnd}`).format.numberFormat = "#,##0.000000";
  storage.getRange(`A2:F${storageEnd}`).format.font = { name: FONT, size: 9, color: "#222222" };

  const emergency = workbook.worksheets.getItem("紧急购电量");
  const emergencyRows = [];
  let priorDate = null;
  for (const row of data.emergencyRows) {
    const date = row.date ?? row["日期"];
    const period = row.period ?? row["紧急购电时间段"];
    const amount = row.amount ?? row["紧急购电量(kWh)"];
    emergencyRows.push([date !== priorDate ? dateValue(date) : null, period, amount]);
    priorDate = date;
  }
  emergency.getRange("A2:C20000").clear({ applyTo: "contents" });
  if (emergencyRows.length) emergency.getRange("A2").write(emergencyRows);
  const emergencyEnd = Math.max(2, emergencyRows.length + 1);
  emergency.getRange(`A2:A${emergencyEnd}`).setNumberFormat("yyyy-mm-dd");
  emergency.getRange(`C2:C${emergencyEnd}`).format.numberFormat = "#,##0.000000";
  emergency.getRange(`A2:C${emergencyEnd}`).format.font = { name: FONT, size: 9, color: "#222222" };
  return { workbook, storageEnd, emergencyEnd };
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const submission42 = await makeSubmission("result4-2.xlsx", bundle.result42, false);
await render(submission42.workbook, "计划购电量", "A1:L8", "4-2_计划购电量.png");
await render(submission42.workbook, "计划购电量", "EN1:EQ8", "4-2_计划合计.png");
await render(submission42.workbook, "充放电量", "A1:F20", "4-2_充放电量.png");
await render(submission42.workbook, "紧急购电量", `A1:C${Math.min(30, submission42.emergencyEnd)}`, "4-2_紧急购电量.png");
const path42 = path.join(outputDir, "result4-2.xlsx");
const errors42 = await exportChecked(submission42.workbook, path42);

const submission43 = await makeSubmission("result4-3.xlsx", bundle.result43, true);
await render(submission43.workbook, "计划购电量", "A1:L8", "4-3_计划购电量.png");
await render(submission43.workbook, "调整购电量", "A1:L8", "4-3_调整购电量.png");
await render(submission43.workbook, "计划购电量", "EN1:EQ8", "4-3_计划合计.png");
await render(submission43.workbook, "充放电量", "A1:F20", "4-3_充放电量.png");
await render(submission43.workbook, "紧急购电量", `A1:C${Math.min(30, submission43.emergencyEnd)}`, "4-3_紧急购电量.png");
const path43 = path.join(outputDir, "result4-3.xlsx");
const errors43 = await exportChecked(submission43.workbook, path43);

const audit = Workbook.create();
const summary = audit.worksheets.add("结果摘要");
const forecast = audit.worksheets.add("电价预测评价");
const daily42 = audit.worksheets.add("问题4-2逐日结果");
const daily43 = audit.worksheets.add("问题4-3逐日结果");
for (const sheet of [summary, forecast, daily42, daily43]) sheet.showGridLines = false;
summary.tabColor = HEADER; forecast.tabColor = GREEN;

summary.getRange("A2").values = [["问题四周期电价预测与联合场景MILP结果"]];
summary.getRange("A2:F2").format.font = { name: FONT, size: 14, bold: true, color: "#222222" };
summary.getRange("A4:C4").values = [["方案", "实际总费用(元)", "紧急购电量(kWh)"]]; header(summary, "A4:C4");
const summaryKeys = ["问题4-2_现实策略", "问题4-2_完全信息价格下界", `问题4-3_S61218_全部四时点`, "问题4-3_完全信息价格下界"];
const dynamicSummaryKeys = Object.keys(bundle.summaries).filter((key) => key.includes("4-2") || key.includes("4-3"));
const summaryRows = dynamicSummaryKeys.filter((key) => bundle.summaries[key]).map((key) => [
  key, bundle.summaries[key]["实际总费用(元)"], bundle.summaries[key]["紧急购电量(kWh)"],
]);
summary.getRange("A5").write(summaryRows); summary.getRange(`B5:C${summaryRows.length + 4}`).format.numberFormat = "#,##0.000000";
summary.getRange("A10:C10").values = [["信息价值", "费用差(元)", "相对差距"]]; header(summary, "A10:C10", GREEN);
const valueRows = ["问题4-2_价格信息价值", "问题4-3_价格信息价值"].filter((key) => bundle.summaries[key]).map((key) => [
  key, bundle.summaries[key]["费用差(元)"], bundle.summaries[key]["相对差距"],
]);
summary.getRange("A11").write(valueRows); summary.getRange("B11:B12").format.numberFormat = "#,##0.000000";
summary.getRange("C11:C12").format.numberFormat = "0.0000%";
summary.getRange("A15:D15").values = [["4-3更新策略", "实际总费用(元)", "相对仅0点节省(元)", "紧急购电量(kWh)"]]; header(summary, "A15:D15", GREEN);
const policyKeys = ["问题4-3_S0_仅0点", "问题4-3_S6_增加6点", "问题4-3_S612_增加6点12点", "问题4-3_S61218_全部四时点"];
const actualPolicyKeys = Object.keys(bundle.summaries).filter((key) => /4-3_S(0|6|612)/.test(key));
const baseCost = bundle.summaries[actualPolicyKeys[0]]["实际总费用(元)"];
const policyRows = actualPolicyKeys.map((key) => [
  key, bundle.summaries[key]["实际总费用(元)"], baseCost - bundle.summaries[key]["实际总费用(元)"],
  bundle.summaries[key]["紧急购电量(kWh)"],
]);
summary.getRange("A16").write(policyRows); summary.getRange("B16:D19").format.numberFormat = "#,##0.000000";
summary.getRange("A:A").format.columnWidth = 36; summary.getRange("B:D").format.columnWidth = 22;

forecast.getRange("A2").values = [["0:00电价预测模型比较"]];
forecast.getRange("A2:G2").format.font = { name: FONT, size: 14, bold: true, color: "#222222" };
const forecastHeaders = ["模型", "MAE(元/kWh)", "RMSE(元/kWh)", "WAPE", "日内Spearman", "前20%高价识别率", "最高价时刻MAE(分钟)"];
forecast.getRange("A4:G4").values = [forecastHeaders]; header(forecast, "A4:G4");
const forecastRows = Object.entries(bundle.forecastMetrics["0点模型比较"]).map(([name, value]) => [
  name, value["MAE(元/kWh)"], value["RMSE(元/kWh)"], value.WAPE, value["日内Spearman"],
  value["前20%高价时段识别精确率"], value["最高价时刻MAE(分钟)"],
]);
forecast.getRange("A5").write(forecastRows);
const forecastEnd = forecastRows.length + 4;
forecast.getRange(`B5:C${forecastEnd}`).format.numberFormat = "0.000000";
forecast.getRange(`D5:F${forecastEnd}`).format.numberFormat = "0.0000%"; forecast.getRange(`G5:G${forecastEnd}`).format.numberFormat = "0.0";
forecast.getRange("A12:D12").values = [["更新时点", "采用更新天数", "周期基线MAE", "门控更新MAE"]]; header(forecast, "A12:D12", GREEN);
const updateRows = Object.entries(bundle.forecastMetrics["日内更新"]).map(([hour, value]) => [
  `${hour}:00`, value["采用更新天数"], value["周期基线MAE(元/kWh)"], value["门控更新MAE(元/kWh)"],
]);
forecast.getRange("A13").write(updateRows); forecast.getRange("C13:D15").format.numberFormat = "0.000000";
forecast.getRange("A18:C18").values = [["滞后天数h", "十分钟滞后阶数144h", "ACF(144h)"]]; header(forecast, "A18:C18", GREEN);
const acfRows = Object.entries(bundle.forecastMetrics["ACF整日滞后"]).map(([lag, value]) => [
  Number(lag), Number(lag) * 144, value,
]);
forecast.getRange("A19").write(acfRows); forecast.getRange("C19:C46").format.numberFormat = "0.000000";
forecast.getRange("A:A").format.columnWidth = 26; forecast.getRange("B:G").format.columnWidth = 21;

function populateDaily(sheet, rows) {
  const filtered = rows.filter((row) => row["日期"] >= "2025-02-01");
  const columns = ["日期", "计划购电量(kWh)", "紧急购电量(kWh)", "正常购电费(元)", "实际总费用(元)", "日初储电量(kWh)", "日末储电量(kWh)"]
    .filter((column) => filtered.length && Object.hasOwn(filtered[0], column));
  sheet.getRangeByIndexes(1, 0, 1, columns.length).values = [columns];
  header(sheet, `A2:${String.fromCharCode(64 + columns.length)}2`);
  const values = filtered.map((row) => columns.map((column) => column === "日期" ? dateValue(row[column]) : row[column]));
  sheet.getRange("A3").write(values);
  sheet.getRange(`A3:A${values.length + 2}`).setNumberFormat("yyyy-mm-dd");
  if (columns.length > 1) sheet.getRange(`B3:${String.fromCharCode(64 + columns.length)}${values.length + 2}`).format.numberFormat = "#,##0.000000";
  sheet.getRange("A:A").format.columnWidth = 14;
  for (let col = 2; col <= columns.length; col += 1) sheet.getRange(`${String.fromCharCode(64 + col)}:${String.fromCharCode(64 + col)}`).format.columnWidth = 20;
  sheet.freezePanes.freezeRows(2); sheet.freezePanes.freezeColumns(1);
}
populateDaily(daily42, bundle.daily42);
populateDaily(daily43, bundle.daily43);

await render(audit, "结果摘要", "A1:D20", "审计_结果摘要.png");
await render(audit, "电价预测评价", "A1:G47", "审计_电价预测评价.png");
await render(audit, "问题4-2逐日结果", "A1:G16", "审计_4-2逐日.png");
await render(audit, "问题4-3逐日结果", "A1:G16", "审计_4-3逐日.png");
const auditPath = path.join(outputDir, "问题四_全年结果与校验.xlsx");
const auditErrors = await exportChecked(audit, auditPath);

const result = {
  outputs: [path42, path43, auditPath], errors42, errors43, auditErrors, previewDir,
  dimensions: {
    result42Plan: `A1:EQ${bundle.result42.purchaseRows.length + 1}`,
    result43Plan: `A1:EQ${bundle.result43.purchaseRows.length + 1}`,
    result43Adjustment: `A1:EQ${bundle.result43.adjustmentRows.length + 1}`,
  },
};
await fs.writeFile(path.join(intermediateDir, "问题四_结果工作簿检查.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
console.log(JSON.stringify(result, null, 2));

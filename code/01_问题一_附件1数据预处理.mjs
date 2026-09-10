import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspace = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(workspace, "附件", "附件1.xlsx");
const templatePath = path.join(workspace, "附件", "附件5", "result1.xlsx");
const outputPath = path.join(workspace, "predata", "问题一_优化模型标准输入.xlsx");
const checkPath = path.join(workspace, "tmp", "问题一_数据质量检查.json");
const inspectSidecarPath = path.join(workspace, "tmp", "问题一_工作簿结构检查.ndjson");
const timeseriesPreviewPath = path.join(workspace, "tmp", "问题一_标准时序数据预览.png");
const parameterPreviewPath = path.join(workspace, "tmp", "问题一_储能参数预览.png");

const artifactToolEntry = process.env.CODEX_ARTIFACT_TOOL_PATH ?? path.join(
  os.homedir(),
  ".cache",
  "codex-runtimes",
  "codex-primary-runtime",
  "dependencies",
  "node",
  "node_modules",
  "@oai",
  "artifact-tool",
  "dist",
  "artifact_tool.mjs",
);

const { FileBlob, SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactToolEntry).href);

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

function pad2(value) {
  return String(value).padStart(2, "0");
}

function minuteLabel(totalMinutes, useNextDayLabel = false) {
  if (totalMinutes === 1440) {
    return useNextDayLabel ? "00:00+1" : "24:00";
  }
  const wrapped = ((totalMinutes % 1440) + 1440) % 1440;
  return `${pad2(Math.floor(wrapped / 60))}:${pad2(wrapped % 60)}`;
}

function normalizeTimeValue(value, fallbackIndex) {
  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return `${pad2(value.getUTCHours())}:${pad2(value.getUTCMinutes())}`;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    const minutes = Math.round((((value % 1) + 1) % 1) * 1440) % 1440;
    return `${pad2(Math.floor(minutes / 60))}:${pad2(minutes % 60)}`;
  }
  if (typeof value === "string" && value.trim()) {
    const raw = value.trim().replace(/：/g, ":");
    if (/^0?0:00\+1$/.test(raw)) return "00:00+1";
    const match = raw.match(/^(\d{1,2}):(\d{2})$/);
    if (match) return `${pad2(Number(match[1]))}:${pad2(Number(match[2]))}`;
    return raw;
  }
  const expectedMinutes = fallbackIndex === 144 ? 1440 : fallbackIndex * 10;
  return minuteLabel(expectedMinutes, fallbackIndex === 144);
}

function requireFiniteNumber(value, fieldName, sourceRow) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`附件1第 ${sourceRow} 行的“${fieldName}”不是有效数值：${String(value)}`);
  }
  return number;
}

function summarize(values) {
  return {
    count: values.length,
    min: Math.min(...values),
    max: Math.max(...values),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    negativeCount: values.filter((value) => value < 0).length,
    zeroCount: values.filter((value) => value === 0).length,
  };
}

function formatTimeseriesSheet(sheet, rowCount, colCount) {
  const lastCol = excelColumnName(colCount);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(2);
  sheet.tabColor = "#1F4E78";

  sheet.getRange(`A1:${lastCol}1`).format.borders = {
    bottom: { style: "thin", color: "#7F8C8D" },
  };
  sheet.getRange("A1").format = {
    font: { name: "Arial", size: 15, bold: true, color: "#000000" },
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastCol}3`).format = {
    font: { name: "Arial", size: 9, italic: true, color: "#595959" },
    verticalAlignment: "center",
  };
  sheet.getRange(`A5:${lastCol}5`).format = {
    fill: "#1F4E78",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#D9E2F3" },
  };
  sheet.getRange(`A6:${lastCol}${5 + rowCount}`).format = {
    font: { name: "Arial", size: 10, color: "#222222" },
    verticalAlignment: "center",
  };
  sheet.getRange(`A6:${lastCol}${5 + rowCount}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#E7E6E6" },
  };

  sheet.getRange(`A6:A${5 + rowCount}`).format.numberFormat = "0";
  sheet.getRange(`G6:G${5 + rowCount}`).format.numberFormat = "0.000000";
  sheet.getRange(`H6:H${5 + rowCount}`).format.numberFormat = "0.0000";
  sheet.getRange(`I6:J${5 + rowCount}`).format.numberFormat = "0.0000";
  sheet.getRange(`K6:M${5 + rowCount}`).format.numberFormat = "0.0000";
  sheet.getRange(`N6:O${5 + rowCount}`).format.numberFormat = "0";

  const widths = [10, 13, 12, 12, 19, 23, 11, 18, 16, 16, 17, 17, 18, 12, 10];
  widths.forEach((width, idx) => {
    const col = excelColumnName(idx + 1);
    sheet.getRange(`${col}:${col}`).format.columnWidth = width;
  });
  sheet.getRange("1:1").format.rowHeight = 25;
  sheet.getRange("5:5").format.rowHeight = 34;
}

function formatParameterSheet(sheet, rowCount) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(5);
  sheet.tabColor = "#5B9BD5";
  sheet.getRange("A1:E1").format.borders = {
    bottom: { style: "thin", color: "#7F8C8D" },
  };
  sheet.getRange("A1").format = {
    font: { name: "Arial", size: 15, bold: true, color: "#000000" },
  };
  sheet.getRange("A2:E3").format = {
    font: { name: "Arial", size: 9, italic: true, color: "#595959" },
  };
  sheet.getRange("A5:E5").format = {
    fill: "#4472C4",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#D9E2F3" },
  };
  sheet.getRange(`A6:E${5 + rowCount}`).format = {
    font: { name: "Arial", size: 10, color: "#222222" },
    verticalAlignment: "center",
  };
  sheet.getRange(`A6:E${5 + rowCount}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#E7E6E6" },
  };
  sheet.getRange("C6:C12").format.numberFormat = "0";
  sheet.getRange("C13:C16").format.numberFormat = "0.0000";
  sheet.getRange("C17:C17").format.numberFormat = "0";
  sheet.getRange("C18:C18").format.numberFormat = "0.000000";
  sheet.getRange("C19:C19").format.numberFormat = "0";
  [43, 26, 19, 16, 55].forEach((width, idx) => {
    const col = excelColumnName(idx + 1);
    sheet.getRange(`${col}:${col}`).format.columnWidth = width;
  });
  sheet.getRange("1:1").format.rowHeight = 25;
  sheet.getRange("5:5").format.rowHeight = 30;
  sheet.getRange(`E6:E${5 + rowCount}`).format.wrapText = true;
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(path.dirname(checkPath), { recursive: true });

const sourceWorkbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));
const sourceSheet = sourceWorkbook.worksheets.getItemAt(0);
const sourceValues = sourceSheet.getRange("A1:D145").values;

if (!Array.isArray(sourceValues) || sourceValues.length !== 145) {
  throw new Error(`附件1应为1行表头和144行数据，实际读取到 ${sourceValues?.length ?? 0} 行。`);
}

const expectedHeaders = ["时间", "电价", "小区负载", "光伏发电预测功率"];
const actualHeaders = sourceValues[0].map((value) => String(value ?? "").trim());
if (JSON.stringify(actualHeaders) !== JSON.stringify(expectedHeaders)) {
  throw new Error(`附件1表头不符合预期：${JSON.stringify(actualHeaders)}`);
}

const templateWorkbook = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
const templateSheet = templateWorkbook.worksheets.getItemAt(0);
const templateValues = templateSheet.getRange("A2:A145").values;
if (!Array.isArray(templateValues) || templateValues.length !== 144) {
  throw new Error(`result1模板应包含144个购电时段，实际读取到 ${templateValues?.length ?? 0} 个。`);
}

const timeSeriesRows = [];
const rawTimes = [];
const prices = [];
const loadsKw = [];
const pvKw = [];
const loadsKwh = [];
const pvKwh = [];
const netLoadsKwh = [];

for (let index = 1; index <= 144; index += 1) {
  const sourceRowNumber = index + 1;
  const row = sourceValues[index];
  if (!Array.isArray(row) || row.length < 4) {
    throw new Error(`附件1第 ${sourceRowNumber} 行字段数量不足。`);
  }

  const rawTime = normalizeTimeValue(row[0], index);
  const price = requireFiniteNumber(row[1], "电价", sourceRowNumber);
  const loadPower = requireFiniteNumber(row[2], "小区负载", sourceRowNumber);
  const pvPower = requireFiniteNumber(row[3], "光伏发电预测功率", sourceRowNumber);
  if (price < 0 || loadPower <= 0 || pvPower < 0) {
    throw new Error(`附件1第 ${sourceRowNumber} 行出现不符合物理边界的值。`);
  }

  const startMinute = (index - 1) * 10;
  const endMinute = index * 10;
  const loadEnergy = loadPower / 6;
  const pvEnergy = pvPower / 6;
  const netLoadEnergy = loadEnergy - pvEnergy;
  const templateLabel = String(templateValues[index - 1]?.[0] ?? "").trim();
  if (!templateLabel) {
    throw new Error(`result1模板第 ${index + 1} 行的时间段为空。`);
  }

  rawTimes.push(rawTime);
  prices.push(price);
  loadsKw.push(loadPower);
  pvKw.push(pvPower);
  loadsKwh.push(loadEnergy);
  pvKwh.push(pvEnergy);
  netLoadsKwh.push(netLoadEnergy);

  timeSeriesRows.push([
    index,
    rawTime,
    minuteLabel(startMinute),
    minuteLabel(endMinute, endMinute === 1440),
    `${minuteLabel(startMinute)}-${minuteLabel(endMinute, endMinute === 1440)}`,
    templateLabel,
    1 / 6,
    price,
    loadPower,
    pvPower,
    loadEnergy,
    pvEnergy,
    netLoadEnergy,
    pvPower > 0 ? 1 : 0,
    sourceRowNumber,
  ]);
}

const expectedRawTimes = Array.from({ length: 144 }, (_, idx) => {
  const index = idx + 1;
  return index === 144 ? "00:00+1" : minuteLabel(index * 10);
});
const timeSequenceMatches = rawTimes.every((value, idx) => value === expectedRawTimes[idx]);
if (!timeSequenceMatches) {
  throw new Error("附件1时间序列不是预期的10分钟连续序列，请检查原始文件。 ");
}

const parameters = [
  ["battery_nominal_capacity_kwh", "储能设备额定容量", 12000, "kWh", "来自题面附录1；实际运行仍受1200至10800 kWh限制"],
  ["soc_min_kwh", "允许的最低储电量", 1200, "kWh", "硬约束下界"],
  ["soc_max_kwh", "允许的最高储电量", 10800, "kWh", "硬约束上界"],
  ["initial_soc_kwh", "0:00初始储电量", 6000, "kWh", "问题一初值"],
  ["terminal_soc_kwh", "24:00目标储电量", 6000, "kWh", "问题一要求与0:00储电量相同"],
  ["max_charge_power_kw", "最大充电功率", 5000, "kW", "来自题面附录1"],
  ["max_discharge_power_kw", "最大放电功率", 5000, "kW", "来自题面附录1"],
  ["max_charge_energy_kwh_per_interval", "单个10分钟时段最大充电量", 5000 / 6, "kWh/时段", "5000 kW乘以1/6小时"],
  ["max_discharge_energy_kwh_per_interval", "单个10分钟时段最大放电量", 5000 / 6, "kWh/时段", "5000 kW乘以1/6小时"],
  ["charge_efficiency", "充电效率", 0.9, "无量纲", "第一版按单程充电效率90%处理"],
  ["discharge_efficiency", "放电效率", 0.9, "无量纲", "第一版按单程放电效率90%处理"],
  ["interval_minutes", "时段长度", 10, "分钟", "附件数据间隔"],
  ["interval_hours", "时段长度", 1 / 6, "小时", "功率换算为电量时使用"],
  ["interval_count", "每日决策时段数", 144, "个", "24小时除以10分钟"],
  ["time_alignment_mode", "模型时间对齐口径", "原始时刻作为区间终点", "文本", "0:10对应模型区间00:00-00:10"],
  ["template_alignment_mode", "结果模板映射口径", "按行序号一一映射", "文本", "同时保留result1模板原始时间段，避免提前覆盖歧义"],
];

const outputWorkbook = Workbook.create();
const timeseriesSheet = outputWorkbook.worksheets.add("标准时序数据");
const parameterSheet = outputWorkbook.worksheets.add("储能参数");

timeseriesSheet.getRange("A1").values = [["问题一优化模型标准输入"]];
timeseriesSheet.getRange("A2").values = [["数据来源：附件/附件1.xlsx；提交时间段来源：附件/附件5/result1.xlsx"]];
timeseriesSheet.getRange("A3").values = [["说明：模型内部按144个10分钟时段计算；负荷和光伏已由kW除以6换算为kWh；原始时刻、模型时段和模板时段均保留。"]];
const timeseriesHeaders = [
  "时段编号",
  "原始时刻",
  "模型起始时刻",
  "模型结束时刻",
  "模型时间段",
  "result1模板时间段",
  "时段长度(h)",
  "外网电价(元/kWh)",
  "小区负荷功率(kW)",
  "光伏预测功率(kW)",
  "小区负荷电量(kWh)",
  "可利用光伏电量(kWh)",
  "净负荷电量(kWh)",
  "光伏是否发电",
  "原附件行号",
];
timeseriesSheet.getRange("A5:O5").values = [timeseriesHeaders];
timeseriesSheet.getRange("A6:O149").values = timeSeriesRows;
timeseriesSheet.tables.add("A5:O149", true, "Problem1TimeseriesTable").style = "TableStyleMedium2";
formatTimeseriesSheet(timeseriesSheet, timeSeriesRows.length, timeseriesHeaders.length);

parameterSheet.getRange("A1").values = [["问题一储能与时间参数"]];
parameterSheet.getRange("A2").values = [["数据来源：C题.pdf附录1；时间换算依据附件1的10分钟间隔"]];
parameterSheet.getRange("A3").values = [["说明：效率暂按充电和放电过程各90%处理；其他解释应在后续灵敏度分析中通过修改参数统一切换。"]];
const parameterHeaders = ["参数名", "中文含义", "参数值", "单位", "使用说明"];
parameterSheet.getRange("A5:E5").values = [parameterHeaders];
parameterSheet.getRange(`A6:E${5 + parameters.length}`).values = parameters;
parameterSheet.tables.add(`A5:E${5 + parameters.length}`, true, "Problem1ParameterTable").style = "TableStyleMedium2";
formatParameterSheet(parameterSheet, parameters.length);

outputWorkbook.recalculate();

const inspection = await outputWorkbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 8000,
  tableMaxRows: 8,
  tableMaxCols: 15,
  tableMaxCellChars: 100,
});

const formulaErrorInspection = await outputWorkbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "问题一预处理工作簿公式错误扫描",
});

const outputBlob = await SpreadsheetFile.exportXlsx(outputWorkbook);
await outputBlob.save(outputPath);

const generatedInspectSidecar = `${outputPath}.inspect.ndjson`;
try {
  await fs.rename(generatedInspectSidecar, inspectSidecarPath);
} catch (error) {
  if (error?.code !== "ENOENT") throw error;
}

const timeseriesPreview = await outputWorkbook.render({
  sheetName: "标准时序数据",
  range: "A1:O25",
  scale: 1.25,
  format: "png",
});
await fs.writeFile(timeseriesPreviewPath, new Uint8Array(await timeseriesPreview.arrayBuffer()));

const parameterPreview = await outputWorkbook.render({
  sheetName: "储能参数",
  range: `A1:E${5 + parameters.length}`,
  scale: 1.5,
  format: "png",
});
await fs.writeFile(parameterPreviewPath, new Uint8Array(await parameterPreview.arrayBuffer()));

const checkReport = {
  task: "问题一附件1数据预处理",
  sourceWorkbook: path.relative(workspace, sourcePath).replaceAll("\\", "/"),
  templateWorkbook: path.relative(workspace, templatePath).replaceAll("\\", "/"),
  outputWorkbook: path.relative(workspace, outputPath).replaceAll("\\", "/"),
  processedAt: new Date().toISOString(),
  rowCount: timeSeriesRows.length,
  headersMatched: true,
  missingValueCount: sourceValues.slice(1).flat().filter((value) => value === null || value === undefined || value === "").length,
  uniqueRawTimeCount: new Set(rawTimes).size,
  timeSequenceMatches,
  timeAlignment: {
    model: "原始时刻作为区间终点；0:10对应00:00-00:10",
    resultTemplate: "按行序号保存result1模板时间段；暂不消除10分钟标签歧义",
  },
  price: summarize(prices),
  loadPowerKw: summarize(loadsKw),
  photovoltaicPowerKw: summarize(pvKw),
  loadEnergyKwh: summarize(loadsKwh),
  photovoltaicEnergyKwh: summarize(pvKwh),
  netLoadEnergyKwh: summarize(netLoadsKwh),
  totals: {
    loadEnergyKwh: loadsKwh.reduce((sum, value) => sum + value, 0),
    photovoltaicEnergyKwh: pvKwh.reduce((sum, value) => sum + value, 0),
  },
  physicalChecks: {
    priceNonnegative: prices.every((value) => value >= 0),
    loadStrictlyPositive: loadsKw.every((value) => value > 0),
    photovoltaicNonnegative: pvKw.every((value) => value >= 0),
    maximumChargeEnergyPerIntervalKwh: 5000 / 6,
    maximumDischargeEnergyPerIntervalKwh: 5000 / 6,
  },
  workbookInspection: inspection.ndjson,
  formulaErrorInspection: formulaErrorInspection.ndjson,
  previewFiles: [
    path.relative(workspace, timeseriesPreviewPath).replaceAll("\\", "/"),
    path.relative(workspace, parameterPreviewPath).replaceAll("\\", "/"),
  ],
};
await fs.writeFile(checkPath, `${JSON.stringify(checkReport, null, 2)}\n`, "utf8");

console.log(JSON.stringify({ outputPath, checkPath, timeseriesPreviewPath, parameterPreviewPath, checkReport }, null, 2));

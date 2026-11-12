// Genera docs/articulo/DengueSense_articulo.docx: el articulo en Word con formato IEEE
// (A4, dos columnas, Times New Roman). Figuras y tablas: scripts/articulo/figuras y tablas.
//
//   python scripts/articulo/generar_material.py
//   cd scripts/articulo && npm install docx && node generar_docx.js
// Para el PDF: abrir el .docx en Word y exportarlo (Archivo > Exportar > PDF).
"use strict";
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, SectionType, HeadingLevel, TabStopType, Tab, VerticalAlign, LevelFormat,
} = require("docx");

const AQUI = __dirname;
const FIG = path.join(AQUI, "figuras");
const TAB = path.join(AQUI, "tablas");
const SALIDA = path.join(AQUI, "..", "..", "docs", "articulo", "DengueSense_articulo.docx");

// ---------------------------------------------------------- pagina IEEE ---
const MM = 56.6929; // DXA por milimetro
const PAGINA = { size: { width: 11906, height: 16838 },
                 margin: { top: Math.round(19 * MM), bottom: Math.round(43 * MM), left: Math.round(14.32 * MM),
                           right: Math.round(14.32 * MM), header: 400, footer: 400 } };
const ANCHO = 11906 - 2 * PAGINA.margin.left;           // ancho de texto (DXA)
const SEP_COL = Math.round(4.22 * MM);
const COL = Math.floor((ANCHO - SEP_COL) / 2);          // ancho de una columna (DXA)
const PX = (dxa) => Math.round((dxa / 1440) * 96);      // DXA -> pixeles (96 ppp)
const FUENTE = "Times New Roman";

// --------------------------------------------------- referencias (IEEE) ---
const REFS = {
  bhatt2013: "S. Bhatt *et al.*, “The global distribution and burden of dengue,” *Nature*, vol. 496, no. 7446, pp. 504–507, 2013.",
  cdcdatos: "Centro Nacional de Epidemiología, Prevención y Control de Enfermedades (CDC-MINSA), “Vigilancia epidemiológica de dengue,” Plataforma Nacional de Datos Abiertos del Perú, 2026. [En línea]. Disponible: https://www.datosabiertos.gob.pe",
  minsaalerta: "CDC-MINSA, “Alerta epidemiológica AE-CDC N.° 006-2024: Dengue en el Perú,” Ministerio de Salud del Perú, Lima, 2024.",
  bortman1999: "M. Bortman, “Elaboración de corredores o canales endémicos mediante planillas de cálculo,” *Rev. Panam. Salud Pública*, vol. 5, no. 1, pp. 1–8, 1999.",
  lowe2021: "R. Lowe *et al.*, “Combined effects of hydrometeorological hazards and urbanisation on dengue risk in Brazil: a spatiotemporal modelling study,” *Lancet Planet. Health*, vol. 5, no. 4, pp. e209–e219, 2021, doi: 10.1016/S2542-5196(20)30292-8.",
  li2026: "D. Li, W. Li, H. Zhang y S. Hu, “Integrating machine learning and data-driven modelling: predicting dengue outbreaks and revealing spatially heterogeneous climatic drivers in Brazil,” *J. R. Soc. Interface*, vol. 23, art. 20260071, 2026, doi: 10.1098/rsif.2026.0071.",
  panja2023: "M. Panja, T. Chakraborty, S. S. Nadim, I. Ghosh, U. Kumar y N. Liu, “An ensemble neural network approach to forecast dengue outbreak based on climatic condition,” *Chaos, Solitons & Fractals*, vol. 167, art. 113124, 2023, doi: 10.1016/j.chaos.2023.113124.",
  nguyen2022: "V.-H. Nguyen *et al.*, “Deep learning models for forecasting dengue fever based on climate data in Vietnam,” *PLoS Negl. Trop. Dis.*, vol. 16, no. 6, 2022.",
  mcgough2021: "S. F. McGough *et al.*, “A dynamic, ensemble learning approach to forecast dengue fever epidemic years in Brazil using weather and population susceptibility cycles,” *J. R. Soc. Interface*, vol. 18, art. 20201006, 2021, doi: 10.1098/rsif.2020.1006.",
  hersbach2020: "H. Hersbach *et al.*, “The ERA5 global reanalysis,” *Q. J. R. Meteorol. Soc.*, vol. 146, no. 730, pp. 1999–2049, 2020.",
  lundberg2017: "S. M. Lundberg y S.-I. Lee, “A unified approach to interpreting model predictions,” en *Advances in Neural Information Processing Systems 30*, 2017, pp. 4765–4774.",
  reiner2019: "R. C. Reiner *et al.*, “Estimating the impact of city-wide *Aedes aegypti* population control: an observational study in Iquitos, Peru,” *PLoS Negl. Trop. Dis.*, vol. 13, art. e0007255, 2019.",
  hochreiter1997: "S. Hochreiter y J. Schmidhuber, “Long short-term memory,” *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.",
  clarke2024: "J. Clarke, A. Lim, P. Gupte, D. M. Pigott, W. G. van Panhuis y O. J. Brady, “A global dataset of publicly available dengue case count data,” *Sci. Data*, vol. 11, art. 296, 2024, doi: 10.1038/s41597-024-03120-7.",
  tashman2000: "L. J. Tashman, “Out-of-sample tests of forecasting accuracy: an analysis and review,” *Int. J. Forecast.*, vol. 16, no. 4, pp. 437–450, 2000.",
  hyndman2021: "R. J. Hyndman y G. Athanasopoulos, *Forecasting: Principles and Practice*, 3.ª ed. Melbourne, Australia: OTexts, 2021.",
  enfen2012: "Comité Multisectorial ENFEN, “Definición operacional de los eventos El Niño y La Niña y sus magnitudes en la costa del Perú,” Nota Técnica ENFEN 01-2012, Lima, 2012.",
  enfen2023: "Comisión Multisectorial ENFEN, “Informe técnico ENFEN, enero de 2023,” IMARPE, Callao, 2023.",
  chen2016: "T. Chen y C. Guestrin, “XGBoost: A scalable tree boosting system,” en *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.",
  paszke2019: "A. Paszke *et al.*, “PyTorch: An imperative style, high-performance deep learning library,” en *Advances in Neural Information Processing Systems 32*, 2019, pp. 8024–8035.",
  pedregosa2011: "F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.",
};
const ORDEN_CITAS = []; // se llena por orden de aparicion
function numeroCita(clave) {
  if (!REFS[clave]) throw new Error("Cita sin referencia: " + clave);
  if (!ORDEN_CITAS.includes(clave)) ORDEN_CITAS.push(clave);
  return ORDEN_CITAS.indexOf(clave) + 1;
}

// -------------------------------------------------- texto con formato -----
// Marcado: **negrita**, *cursiva*, _{subindice}, ^{superindice}, {c:clave1,clave2} = citas
function runs(texto, base = {}) {
  texto = texto.replace(/__TAB_(\w+)__/g, (_, k) => (ETIQ_TAB[k] ? tab(k) : "Tabla ?"))
               .replace(/__FIG_(\w+)__/g, (_, k) => (ETIQ_FIG[k] ? fig(k) : "Fig. ?"));
  const salida = [];
  const patron = /\*\*(.+?)\*\*|\*(.+?)\*|_\{(.+?)\}|\^\{(.+?)\}|\{c:([^}]+)\}/g;
  let ultimo = 0, m;
  const plano = (t, extra = {}) => { if (t) salida.push(new TextRun({ text: t, font: FUENTE, ...base, ...extra })); };
  while ((m = patron.exec(texto))) {
    plano(texto.slice(ultimo, m.index));
    if (m[1] !== undefined) salida.push(...runs(m[1], { ...base, bold: true }));
    else if (m[2] !== undefined) salida.push(...runs(m[2], { ...base, italics: !base.italics }));
    else if (m[3] !== undefined) plano(m[3], { subScript: true });
    else if (m[4] !== undefined) plano(m[4], { superScript: true });
    else if (m[5] !== undefined) plano(m[5].split(",").map((c) => `[${numeroCita(c.trim())}]`).join(", "));
    ultimo = patron.lastIndex;
  }
  plano(texto.slice(ultimo));
  return salida;
}
const P = (texto, opciones = {}) => new Paragraph({ children: runs(texto), alignment: AlignmentType.JUSTIFIED,
                                                     indent: { firstLine: 180 }, ...opciones });
const P0 = (texto, opciones = {}) => P(texto, { indent: { firstLine: 0 }, ...opciones });
const VINETA = (texto) => new Paragraph({ children: runs(texto), alignment: AlignmentType.JUSTIFIED,
                                          numbering: { reference: "vinetas", level: 0 } });
const NUM = (texto, ref = "lista") => new Paragraph({ children: runs(texto), alignment: AlignmentType.JUSTIFIED,
                                                      numbering: { reference: ref, level: 0 } });
const ROMANOS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
let nSec = 0, nSub = 0;
function H1(texto) {
  nSub = 0;
  const etiqueta = texto === "Referencias" ? "" : `${ROMANOS[nSec++]}. `;
  return new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
                         children: [new TextRun({ text: etiqueta + texto, font: FUENTE, smallCaps: true })] });
}
function H2(texto) {
  const letra = String.fromCharCode(65 + nSub++);
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(`${letra}. ${texto}`, { italics: true }) });
}
const PARRAFO_TITULADO = (titulo, texto) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 180 },
  children: [...runs(titulo + ": ", { italics: true }), ...runs(texto)] });

// ---------------------------------------------------------- figuras -------
let nFig = 0;
const ETIQ_FIG = {};
function tamPNG(archivo) {
  const b = fs.readFileSync(archivo);
  return { b, w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figura(clave, archivo, anchoDxa, pie) {
  ETIQ_FIG[clave] = ++nFig;
  const { b, w, h } = tamPNG(path.join(FIG, archivo));
  const ancho = PX(anchoDxa);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
                    children: [new ImageRun({ type: "png", data: b, transformation: { width: ancho, height: Math.round((ancho * h) / w) },
                                              altText: { title: `Fig. ${nFig}`, description: pie.replace(/[*_{}^]/g, ""), name: archivo } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
                    children: [new TextRun({ text: `Fig. ${nFig}. `, font: FUENTE, size: 16 }), ...runs(pie, { size: 16 })] }),
  ];
}
const fig = (clave) => `Fig. ${ETIQ_FIG[clave]}`;

// ----------------------------------------------------------- tablas -------
let nTab = 0;
const ETIQ_TAB = {};
const BORDE = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
const FINO = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const NADA = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function celda(texto, ancho, { negrita = false, alinear = AlignmentType.CENTER, filas = 1, columnas = 1, arriba = NADA, abajo = NADA } = {}) {
  return new TableCell({
    width: { size: ancho, type: WidthType.DXA }, rowSpan: filas, columnSpan: columnas, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    borders: { top: arriba, bottom: abajo, left: NADA, right: NADA },
    children: [new Paragraph({ alignment: alinear, children: runs(texto, { size: 15, bold: negrita }) })],
  });
}
function tabla(clave, titulo, anchos, encabezados, filas) {
  ETIQ_TAB[clave] = ++nTab;
  const total = anchos.reduce((a, b) => a + b, 0);
  const filasEnc = encabezados.map((fila, k) => new TableRow({ tableHeader: true, children: fila.map((c) =>
    celda(c.t, c.w ?? anchos[c.i], { negrita: false, columnas: c.span || 1, alinear: c.izq ? AlignmentType.LEFT : AlignmentType.CENTER,
                                    arriba: k === 0 ? BORDE : NADA, abajo: k === encabezados.length - 1 ? FINO : (c.linea ? FINO : NADA) })) }));
  const cuerpo = filas.map((fila, r) => new TableRow({ children: fila.celdas.map((c) =>
    celda(c.t, anchos[c.i], { negrita: c.b, filas: c.filas || 1, alinear: c.i <= (fila.izq ?? 0) ? AlignmentType.LEFT : AlignmentType.CENTER,
                              arriba: fila.separar ? FINO : NADA, abajo: r === filas.length - 1 || (c.filas && r + c.filas === filas.length) ? BORDE : NADA })) }));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160 }, keepNext: true,
                    children: [new TextRun({ text: `TABLA ${ROMANOS[nTab - 1]}`, font: FUENTE, size: 16, smallCaps: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, keepNext: true, children: runs(titulo, { size: 16, smallCaps: true }) }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: anchos, alignment: AlignmentType.CENTER,
                borders: { top: NADA, bottom: NADA, left: NADA, right: NADA, insideHorizontal: NADA, insideVertical: NADA },
                rows: [...filasEnc, ...cuerpo] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
const tab = (clave) => `Tabla ${ROMANOS[ETIQ_TAB[clave] - 1]}`;

// Lee las tablas .tex que genera generar_material.py (una sola fuente de numeros)
function leerTex(nombre) {
  const grupos = [];
  let grupo = [];
  for (const linea of fs.readFileSync(path.join(TAB, nombre), "utf8").split(/\r?\n/)) {
    const l = linea.trim();
    if (!l) continue;
    if (l === "\\hline") { grupos.push(grupo); grupo = []; continue; }
    const celdas = l.replace(/\\\\\s*$/, "").split("&").map((c) => {
      let t = c.trim(), negrita = false, filas = 1;
      const mr = t.match(/^\\multirow\{(\d+)\}\{\*\}\{(.*)\}$/);
      if (mr) { filas = +mr[1]; t = mr[2]; }
      const bf = t.match(/^\\textbf\{(.*)\}$/);
      if (bf) { negrita = true; t = bf[1]; }
      t = t.replace(/\$-\$/g, "−").replace(/\\,/g, " ");
      return { t, b: negrita, filas };
    });
    grupo.push(celdas);
  }
  if (grupo.length) grupos.push(grupo);
  return grupos;
}
// Convierte grupos leidos en filas, quitando las celdas cubiertas por un rowSpan
function filasDe(grupos, izq = 1) {
  const filas = [];
  grupos.forEach((g, gi) => g.forEach((celdas, ri) => {
    const conIndice = celdas.map((c, i) => ({ ...c, i }));
    const visibles = conIndice.filter((c, i) => !(ri > 0 && i === 0 && g[0][0].filas > 1));
    filas.push({ celdas: visibles.map((c) => (c.filas > 1 ? c : { ...c, filas: undefined })), separar: gi > 0 && ri === 0, izq });
  }));
  return filas;
}

// ------------------------------------------------------ ecuaciones --------
function ecuacion(partes, numero) {
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(COL / 2) }, { type: TabStopType.RIGHT, position: COL }],
    spacing: { before: 100, after: 100 },
    children: [new TextRun({ children: [new Tab()], font: FUENTE }), ...partes,
               new TextRun({ children: [new Tab(), `(${numero})`], font: FUENTE })],
  });
}
const m = (t, o = {}) => new TextRun({ text: t, font: FUENTE, ...o });
const mi = (t, o = {}) => m(t, { italics: true, ...o });
const sub = (t) => m(t, { subScript: true, italics: true });
const sup = (t) => m(t, { superScript: true });

// ====================================================== CONTENIDO =========
function construirContenido() {
nSec = 0; nSub = 0; nFig = 0; nTab = 0;
const C1 = [];   // seccion de una columna (titulo)
const nl = " ";

C1.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: "DengueSense: pronóstico semanal de dengue y alertas de brote por departamento en el Perú con aprendizaje automático, clima y El Niño costero", font: FUENTE, size: 48 })] }));
// Bloques de autores al estilo IEEE: uno por integrante, lado a lado
const AUTORES = [
  ["Adriano Jesús Medrano Garay", "adriano.medrano@unmsm.edu.pe"],
  ["Julio Daniel Chavez Huamani", "julio.chavezh@unmsm.edu.pe"],
  ["Oder Pedro Quincho Cordova", "oder.quincho@unmsm.edu.pe"],
];
const anchoAutor = Math.floor(ANCHO / AUTORES.length);
const lineaAutor = (texto, opciones) => new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: texto, font: FUENTE, ...opciones })] });
C1.push(new Table({
  width: { size: anchoAutor * AUTORES.length, type: WidthType.DXA }, columnWidths: AUTORES.map(() => anchoAutor),
  borders: { top: NADA, bottom: NADA, left: NADA, right: NADA, insideHorizontal: NADA, insideVertical: NADA },
  rows: [new TableRow({ children: AUTORES.map(([nombre, correo]) => new TableCell({
    width: { size: anchoAutor, type: WidthType.DXA }, borders: { top: NADA, bottom: NADA, left: NADA, right: NADA },
    margins: { left: 80, right: 80 },
    children: [
      lineaAutor(nombre, { size: 22 }),
      lineaAutor("Facultad de Ingeniería de Sistemas e Informática", { size: 18, italics: true }),
      lineaAutor("Universidad Nacional Mayor de San Marcos", { size: 18, italics: true }),
      lineaAutor("Lima, Perú", { size: 18 }),
      lineaAutor(correo, { size: 18 }),
    ] })) })],
}));
C1.push(new Paragraph({ spacing: { after: 200 }, children: [] }));

// --------------------------------------------- dos columnas: parte 1 -----
const A = [];
A.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: [
  new TextRun({ text: "Resumen", font: FUENTE, size: 18, bold: true, italics: true }), new TextRun({ text: "—", font: FUENTE, size: 18, bold: true }),
  ...runs(`El dengue alcanzó en el Perú cifras récord en 2023 y 2024, y la respuesta sanitaria sigue siendo mayormente reactiva. Este trabajo presenta DengueSense, un sistema que pronostica los casos semanales de dengue por departamento con 1 a 4 semanas de anticipación y emite alertas de brote basadas en el canal endémico. Se integraron 1${nl}065${nl}274 notificaciones del CDC-MINSA (2000–2025), clima semanal en 181 celdas de grilla, el Índice Costero El Niño (ICEN) y la población del INEI, en un panel de 23 departamentos y 782 semanas. Se compararon la persistencia, un promedio estacional, una regresión con casos rezagados, XGBoost con y sin clima, y un LSTM, bajo una partición estrictamente temporal y explicando el modelo con SHAP. Aprender el *cambio* de casos respecto a la última semana conocida, en lugar de su nivel, fue clave para que XGBoost superara a la persistencia. Con una validación de origen móvil de nueve años (2016–2024), el clima no mejoró el pronóstico de forma consistente: redujo el error en solo 8 a 11 de 17 departamentos. Sin embargo, los factores climáticos que usa el modelo difieren entre la costa norte (temperatura) y la selva (humedad, lluvia y viento). En la prueba fuera de muestra de 2025, el modelo final superó a la persistencia en 11 a 13 de 17 departamentos. Anticipar el *inicio* de un brote sigue siendo difícil: ningún modelo superó el 21${nl}% a dos semanas.`, { size: 18, bold: true })] }));
A.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 }, children: [
  new TextRun({ text: "Palabras clave", font: FUENTE, size: 18, bold: true, italics: true }), new TextRun({ text: "—", font: FUENTE, size: 18, bold: true }),
  new TextRun({ text: "dengue, pronóstico, aprendizaje automático, XGBoost, SHAP, LSTM, El Niño costero, vigilancia epidemiológica, Perú.", font: FUENTE, size: 18, bold: true })] }));

A.push(H1("Introducción"));
A.push(P(`El dengue es la arbovirosis de mayor carga en el mundo, con cientos de millones de infecciones al año {c:bhatt2013}. En el Perú, los datos abiertos de vigilancia del CDC-MINSA registran 256${nl}641 casos en 2023 y 271${nl}531 en 2024, las mayores cifras desde 2000 (Fig. 1) {c:cdcdatos}. Según el propio Minsa, en 2023 se notificaron 273${nl}684 casos, el 84,8${nl}% de ellos confirmados {c:minsaalerta}. La transmisión, antes concentrada en la costa norte y la Amazonía, se extendió a Lima Metropolitana, donde en 2024 se registraron 87${nl}864 casos.`));
A.push(P(`La respuesta suele ser reactiva: se fumiga y se refuerzan los servicios cuando los casos ya superaron el canal endémico {c:bortman1999}. Un pronóstico de corto plazo por región permitiría actuar antes del pico. La literatura muestra que el clima modula la transmisión del vector *Aedes aegypti*, con efectos rezagados de la lluvia y la temperatura {c:lowe2021}. También muestra que el aprendizaje automático puede pronosticar el dengue y revelar factores climáticos que cambian según la región {c:li2026,panja2023,nguyen2022,mcgough2021}.`));
A.push(P(`Este trabajo adapta al Perú el enfoque de Li *et al.* {c:li2026}, que combina XGBoost y SHAP, y plantea tres hipótesis:`));
A.push(VINETA("**H1:** los factores climáticos más influyentes difieren entre la costa y la selva."));
A.push(VINETA("**H2:** un modelo con clima rezagado supera a uno que solo usa los casos pasados y la estacionalidad."));
A.push(VINETA("**H3:** el desempeño es peor en periodos atípicos."));
A.push(P0("Las contribuciones son:"));
A.push(NUM("un panel reproducible departamento × semana que integra casos, clima, ICEN y población, con controles de fuga de información;", "contribuciones"));
A.push(NUM("una evaluación honesta con nueve años de prueba (origen móvil), métricas de alerta de brote y una prueba real fuera de muestra en 2025;", "contribuciones"));
A.push(NUM("el análisis de Lima como caso de dengue emergente;", "contribuciones"));
A.push(NUM("una interfaz de libre acceso con mapas de riesgo.", "contribuciones"));

// --------------------------------------------- figura ancha 1 ------------
const W1 = figura("anuales", "casos_anuales.png", Math.round(ANCHO * 0.92), "Casos de dengue notificados por año en el Perú, 2000–2025 (datos abiertos del CDC-MINSA; 2025 provisional). En naranja, los años epidémicos 2017, 2023 y 2024.");

// --------------------------------------------- dos columnas: parte 2 -----
const B = [];
B.push(H1("Trabajos relacionados"));
B.push(P(`Li *et al.* {c:li2026} pronosticaron el dengue semanal por estado en Brasil (2020–2025) con XGBoost alimentado por el clima de ERA5 {c:hersbach2020} y los casos rezagados. Conservaron una variable por categoría climática y usaron SHAP {c:lundberg2017} para ordenar los factores. El R^{2} de prueba osciló entre 0,50 y 0,95 según el estado. McGough *et al.* {c:mcgough2021} anticiparon años epidémicos en Brasil con aprendizaje por ensambles. Lowe *et al.* {c:lowe2021} cuantificaron los efectos rezagados de sequías y lluvias extremas en 558 microrregiones brasileñas. En Iquitos (Perú), Panja *et al.* {c:panja2023} propusieron una red neuronal de ondículas con variables climáticas, y Reiner *et al.* {c:reiner2019} evaluaron el control del vector. Nguyen *et al.* {c:nguyen2022} compararon modelos de aprendizaje profundo, entre ellos el LSTM {c:hochreiter1997}, en Vietnam. Para validar los conteos se usó OpenDengue {c:clarke2024}.`));
B.push(P(`A diferencia de esos trabajos, aquí: (i) se cubren los 17 departamentos peruanos con suficientes casos; (ii) se exige que toda variable esté disponible al momento de predecir; y (iii) se evalúa con varios cortes temporales {c:tashman2000,hyndman2021}, porque un solo corte puede llevar a conclusiones distintas, como se muestra en la Sección V-B.`));

B.push(H1("Datos"));
B.push(P(`La __TAB_datos__ resume las fuentes.`));
B.push(VINETA(`**Casos.** Cada notificación incluye la semana epidemiológica de inicio de síntomas y el ubigeo del lugar probable de infección {c:cdcdatos}.`));
B.push(VINETA(`**Clima.** La tabla semanal por celda de grilla (temperaturas, precipitación, días de lluvia, humedad relativa y viento) sigue el mismo calendario epidemiológico que los casos. Se verificó que los años con semana 53 coinciden.`));
B.push(VINETA(`**ICEN.** Es la media móvil de tres meses de la anomalía de la temperatura del mar en la región Niño 1+2 {c:enfen2012,enfen2023}.`));
B.push(VINETA(`**Población.** Proyecciones del INEI por distrito para 2018–2026. Los años 2010–2017 se estimaron hacia atrás con el crecimiento anual de cada departamento.`));
const T_DATOS = tabla("datos", "Fuentes de datos", [1400, 1800, COL - 3200],
  [[{ t: "Fuente", i: 0, izq: true }, { t: "Unidad y periodo", i: 1, izq: true }, { t: "Uso", i: 2, izq: true }]],
  [
    { celdas: [{ t: "CDC-MINSA {c:cdcdatos}", i: 0 }, { t: "Caso individual, 2000–2025", i: 1 }, { t: "Variable objetivo (casos por semana)", i: 2 }], izq: 2 },
    { celdas: [{ t: "Clima por celda", i: 0 }, { t: "181 celdas × 782 semanas, 2010–2024", i: 1 }, { t: "Predictores rezagados", i: 2 }], izq: 2 },
    { celdas: [{ t: "ICEN (ENFEN) {c:enfen2012}", i: 0 }, { t: "Mes, 2009–2024", i: 1 }, { t: "Predictor de El Niño costero", i: 2 }], izq: 2 },
    { celdas: [{ t: "INEI", i: 0 }, { t: "Distrito × año, 2018–2026", i: 1 }, { t: `Incidencia por 100${nl}000 hab.`, i: 2 }], izq: 2 },
    { celdas: [{ t: "OpenDengue {c:clarke2024}", i: 0 }, { t: "Provincia y país, 2000–2024", i: 1 }, { t: "Validación externa", i: 2 }], izq: 2 },
  ]);
B.push(...T_DATOS);
B.push(H2("Limpieza y control de calidad"));
B.push(VINETA(`**Sin identificador de caso.** El archivo no tiene un ID de paciente, así que las 152${nl}091 filas idénticas (14,8${nl}%) se marcaron y se conservaron.`));
B.push(VINETA(`**Edades y nombres.** Se anularon 14 edades imposibles. La letra Ñ, dañada en el archivo fuente, se restauró en provincias y distritos.`));
B.push(VINETA(`**Fuga en el clima.** Se comprobó que las anomalías climáticas de la tabla se calcularon con todos los años (2010–2024), incluidos los de prueba. Por esa fuga de información se excluyeron.`));
B.push(VINETA(`**Validación externa.** El archivo coincide con OpenDengue {c:clarke2024} por provincia. Frente a los totales nacionales oficiales tiene 6,4${nl}% menos casos en 2023 y 3,3${nl}% menos en 2024. En años de pocos casos la diferencia llega al 32${nl}%.`));
B.push(H2("Panel departamento × semana"));
B.push(P(`Los casos se agregaron por departamento según el ubigeo, con 937${nl}119 casos en 2010–2024 y 17${nl}986 filas (23 departamentos × 782 semanas). El clima de cada departamento es el promedio de sus celdas *ponderado por los casos de entrenamiento* (2010–2021). Con un promedio simple, Piura resultaba 2,2${nl}°C más fría y 5 puntos más húmeda que la zona donde ocurren los casos, porque se mezclaban celdas de sierra sin dengue. La __FIG_mapa__ muestra la incidencia media.`));
B.push(P(`Se modelaron los 17 departamentos con al menos 1${nl}000 casos en el periodo de entrenamiento. Lima se excluyó del modelo principal, porque el 96${nl}% de sus casos de 2010–2024 ocurrió en 2023–2024, y se analiza aparte (Sección V-E).`));
B.push(...figura("mapa", "mapa_incidencia.png", Math.round(COL * 0.85), "Incidencia media anual de dengue por departamento, 2010–2024 (escala logarítmica). Con borde negro, los 17 departamentos modelados."));

B.push(H1("Metodología"));
B.push(P(`La __FIG_flujo__ resume el flujo de trabajo.`));
B.push(...figura("flujo", "flujo.png", COL, "Flujo de trabajo de DengueSense."));
B.push(H2("Problema de pronóstico"));
B.push(P(`Sea *y*_{d,t} el número de casos del departamento *d* en la semana epidemiológica *t*. Para cada horizonte *h* ∈ {1, 2, 3, 4} se pronostica *y*_{d,t} usando solo información disponible en la semana *t* − *h*:`));
B.push(VINETA(`casos de *t* − *h* a *t* − *h* − 3;`));
B.push(VINETA(`clima de *t* − *h* a *t* − *h* − 3 (hasta 12 semanas en la prueba de sensibilidad);`));
B.push(VINETA(`el ICEN conocido en *t* − *h*, con al menos tres meses de rezago, porque el ENFEN lo publica con demora {c:enfen2023};`));
B.push(VINETA(`el seno y el coseno de la semana objetivo.`));
B.push(P(`Como en {c:li2026}, dentro de cada categoría climática (temperatura, lluvia, humedad, viento e ICEN) se conserva la variable con mayor correlación de Spearman con log(1 + *y*). La correlación se calcula solo con los datos de entrenamiento.`));
B.push(H2("Modelos"));
B.push(VINETA(`**Persistencia:** ŷ_{d,t} = *y*_{d,t−h}.`));
B.push(VINETA(`**Promedio estacional:** la media de la misma semana en los años anteriores.`));
B.push(VINETA(`**Regresión lineal** de log(1 + *y*_{d,t}) sobre los casos rezagados.`));
B.push(VINETA(`**XGBoost** {c:chen2016}, uno por departamento y horizonte. Hay que notar que los árboles no pueden pronosticar niveles fuera del rango de entrenamiento: en San Martín 2024 el modelo quedaba fijo en 80–150 casos semanales con 190–380 reales. Por eso, además del nivel (log o Poisson), se probó como objetivo el cambio:`));
B.push(ecuacion([mi("Δ"), sup("(h)"), sub("d,t"), m(" = log(1 + "), mi("y"), sub("d,t"), m(") − log(1 + "), mi("y"), sub("d,t−h"), m(")")], 1));
B.push(P0(`del que se recupera ŷ_{d,t} = exp(log(1 + *y*_{d,t−h}) + Δ̂^{(h)}_{d,t}) − 1. La validación eligió este objetivo en el 92${nl}% de los modelos.`, { indent: { left: 360 } }));
B.push(VINETA(`**LSTM** {c:hochreiter1997}: una red global por horizonte, entrenada con los 17 departamentos, una ventana de 12 semanas, un *embedding* del departamento y el mismo objetivo (1), implementada en PyTorch {c:paszke2019}.`));
B.push(P(`Las líneas base se ajustaron con scikit-learn {c:pedregosa2011}.`));
B.push(H2("Evaluación"));
B.push(VINETA(`**Partición fija:** entrenamiento 2010–2021, validación 2022–2023 (hiperparámetros y parada temprana) y prueba 2024, con reentrenamiento en 2010–2023.`));
B.push(VINETA(`**Origen móvil** {c:tashman2000,hyndman2021}: para cada año *Y* ∈ {2016, …, 2024}, las variables y los hiperparámetros se eligen con los años ≤ *Y* − 2, se valida en *Y* − 1, se reentrena con ≤ *Y* − 1 y se evalúa en *Y*.`));
B.push(VINETA(`**Prueba fuera de muestra:** 2025, que no se usó en ninguna etapa.`));
B.push(P(`Las métricas son MAE, RMSE y R^{2} en la escala de casos.`));
B.push(PARRAFO_TITULADO("Definición de brote", `Siguiendo el canal endémico {c:bortman1999}, la semana es de brote si`));
B.push(ecuacion([mi("b"), sub("d,t"), m(" = 1[ "), mi("y"), sub("d,t"), m(" > Q"), m("3", { subScript: true }), m("({"), mi("y"), sub("d,s,a"), m("}"), sub("a∈A"), m("t", { subScript: true, italics: true }), m(") ∧ "), mi("y"), sub("d,t"), m(" ≥ 5 ]")], 2));
B.push(P0(`donde Q_{3} es el percentil 75 de la misma semana *s* en los siete años previos no epidémicos (se excluyen 2017, 2023 y 2024). El modelo emite una alerta si ŷ_{d,t} supera ese umbral. Se reportan la precisión, el *recall*, las falsas alarmas por departamento y año, y el porcentaje de *brotes nuevos* anticipados, es decir, sin brote en las cuatro semanas previas.`));
B.push(PARRAFO_TITULADO("Explicabilidad", `Se usa SHAP (TreeExplainer) {c:lundberg2017} sobre el XGBoost con clima. La importancia media |φ| se agrupa por categoría.`));

B.push(H1("Resultados"));
B.push(H2("Partición fija: prueba 2024"));
B.push(P(`La __TAB_r2__ y la __FIG_regiones__ comparan los modelos en las tres regiones de estudio: Piura (costa norte), Loreto (selva baja) y San Martín (selva alta).`));
B.push(VINETA(`**Inercia de los casos.** A una semana, la persistencia ya alcanza un R^{2} de 0,83 a 0,94: la inercia de los casos domina el corto plazo.`));
B.push(VINETA(`**Con más anticipación.** El R^{2} cae, y ahí XGBoost y el LSTM con clima aportan más.`));
B.push(VINETA(`**Piura.** El LSTM con clima llega a R^{2} = 0,89 y 0,87 a 3 y 4 semanas.`));
B.push(VINETA(`**San Martín.** Ningún modelo supera claramente a la persistencia.`));
B.push(VINETA(`**Promedio estacional.** Fracasó en 2024 (R^{2} ≤ 0,01), porque fue un año atípico.`));
const H_ = ["*h* = 1", "*h* = 2", "*h* = 3", "*h* = 4"];
B.push(...tabla("r2", "R^{2} en la prueba 2024 (en negrita, el mejor por región y horizonte)", [1000, 1621, 600, 600, 600, 600],
  [[{ t: "Región", i: 0, izq: true }, { t: "Modelo", i: 1, izq: true }, ...H_.map((t, k) => ({ t, i: 2 + k }))]],
  filasDe(leerTex("r2_prueba_2024.tex"), 1)));

// --------------------------------------------- figura ancha 2 ------------
const W2 = figura("regiones", "fase3_comparacion_regiones.png", ANCHO, "R^{2} en la prueba 2024 según la anticipación del pronóstico, en las tres regiones de estudio.");

// --------------------------------------------- dos columnas: parte 3 -----
const Cc = [];
Cc.push(H2("¿El clima mejora el pronóstico? (H2)"));
Cc.push(P(`Con la partición fija, la respuesta depende del corte:`));
Cc.push(VINETA(`**Validación 2022–2023:** el XGBoost con clima redujo el MAE frente al sin clima en 16 de 17 departamentos a 4 semanas.`));
Cc.push(VINETA(`**Prueba 2024:** solo lo redujo en 6 de 17.`));
Cc.push(P(`La validación de origen móvil (__TAB_h2__ y __FIG_h2__) resuelve la contradicción. Año a año, el clima gana en alrededor de la mitad de los departamentos, lo que se esperaría si no aportara información. En el error acumulado de 2016–2024, la mejora mediana está entre −1,3${nl}% y +1,6${nl}%. Los años 2022 y 2023 fueron favorables al clima, lo que explica el optimismo de la primera validación.`));
Cc.push(P(`En cambio, XGBoost sí redujo el error de la persistencia en 9 a 13 de 17 departamentos. Por ejemplo, en Piura a dos semanas el MAE bajó de 146 a 108 casos por semana.`));
Cc.push(...tabla("h2", "Validación de origen móvil, 2016–2024: XGBoost con clima frente a sin clima", [500, 1130, 1130, 1130, COL - 3890],
  [[{ t: "*h*", i: 0 }, { t: "Deptos. donde el clima mejora", i: 1 }, { t: "Mejora mediana (%)", i: 2 }, { t: "Años con mayoría a favor", i: 3 }, { t: "XGBoost sin clima mejor que persist.", i: 4 }]],
  filasDe(leerTex("h2_origen_movil.tex"), -1)));
Cc.push(...figura("h2", "fase5_h2_por_anio.png", COL, "Departamentos (de 17) en los que el clima reduce el error, por año evaluado y horizonte. La línea discontinua marca la mitad."));
Cc.push(H2("Alertas de brote"));
Cc.push(P(`A nivel departamental, el 44,8${nl}% de las semanas evaluadas superó el canal endémico. Las alertas de todos los modelos tienen una precisión y un *recall* cercanos al 90${nl}% (__TAB_alertas__), porque los brotes duran varias semanas y avisar de un brote que *continúa* es fácil.`));
Cc.push(P(`Lo difícil es anticipar los 134 *brotes nuevos*. A dos semanas, ningún modelo anticipa más del 21${nl}% (__FIG_alertas__). El clima aumenta este porcentaje en XGBoost (de 16,4 a 20,2${nl}% a dos semanas), pero sin superar a la persistencia, y con tan pocos brotes la diferencia puede deberse al azar.`));
Cc.push(...tabla("alertas", "Alertas de brote, 17 departamentos, 2016–2024 (%)", [380, 1541, 750, 750, 800, COL - 4221],
  [[{ t: "*h*", i: 0 }, { t: "Modelo", i: 1, izq: true }, { t: "Precisión", i: 2 }, { t: "*Recall*", i: 3 }, { t: "Falsas al./ depto.-año", i: 4 }, { t: "Brotes nuevos anticipados", i: 5 }]],
  filasDe(leerTex("alertas.tex"), 1)));

const W3 = figura("alertas", "fase5_alertas.png", ANCHO, "Desempeño de las alertas de brote por modelo y horizonte (17 departamentos, 2016–2024).");

const Dd = [];
Dd.push(H2("Factores climáticos por región (H1)"));
Dd.push(P(`SHAP confirma que los casos pasados y la estacionalidad explican la mayor parte del pronóstico. Dentro del aporte climático, sin embargo, hay un patrón regional claro (__FIG_shap__):`));
Dd.push(VINETA(`**Costa norte:** domina la temperatura (51${nl}% del aporte climático en Piura, 65${nl}% en Lambayeque y 47${nl}% en Ica).`));
Dd.push(VINETA(`**Selva:** pesan más la humedad (40${nl}% en Ucayali y 35${nl}% en Madre de Dios), la lluvia y el viento.`));
Dd.push(VINETA(`**Sierra y selva alta:** el viento destaca, probablemente como marcador estacional más que como causa.`));
Dd.push(P(`Estos resultados son asociaciones que usa el modelo, no relaciones causales.`));
Dd.push(...figura("shap", "fase3_shap_categorias.png", Math.round(COL * 0.95), "Peso relativo de cada categoría climática (SHAP, pronóstico a dos semanas) por departamento."));
Dd.push(H2("Lima: dengue emergente (H3)"));
Dd.push(P(`Lima pasó de 938 casos en 2022 a 30${nl}735 en 2023 y 87${nl}864 en 2024. El 86${nl}% de ellos ocurrió en Lima Metropolitana, y 36 de sus 79 distritos registraron casos por primera vez.`));
Dd.push(P(`La emergencia coincidió con El Niño costero de 2023: el ICEN llegó a 2,9, la temperatura subió 1,05${nl}°C y la humedad 3,5 puntos sobre la media de 2010–2022 en las mismas semanas (__FIG_lima__).`));
Dd.push(P(`Un XGBoost entrenado solo con la historia de Lima fue el peor modelo en 2023 (__TAB_lima__). Uno entrenado con seis departamentos costeros redujo el error un 30${nl}% frente a la persistencia, lo que confirma H3 y muestra el valor de transferir modelos que aprenden el cambio (1).`));
Dd.push(...figura("lima", "lima_clima_2022_2024.png", COL, "Lima: casos semanales, anomalía de temperatura e ICEN, 2022–2024."));
Dd.push(...tabla("lima", "Lima: MAE del pronóstico a dos semanas (casos/semana)", [COL - 2000, 1000, 1000],
  [[{ t: "Modelo", i: 0, izq: true }, { t: "2023", i: 1 }, { t: "2024", i: 2 }]],
  filasDe(leerTex("lima.tex"), 0)));
Dd.push(H2("Prueba fuera de muestra 2025, LSTM y pronóstico"));
Dd.push(P(`Los modelos finales se entrenaron con 2010–2024 y se aplicaron a 2025, un año de pocos casos. El XGBoost sin clima superó a la persistencia en 11 a 13 de 17 departamentos (__TAB_oos__, __FIG_oos__).`));
Dd.push(P(`En Piura falló a 3 y 4 semanas: anticipó la subida estacional aprendida de los años epidémicos, que en 2025 no ocurrió. Es un riesgo propio de entrenar con años atípicos.`));
Dd.push(P(`El LSTM con clima obtuvo el menor MAE mediano en 2024. Sin embargo, superó al XGBoost sin clima en solo 5 a 6 de 17 departamentos a 3 y 4 semanas, por lo que se le considera una línea prometedora y no un reemplazo.`));
Dd.push(P(`Para las semanas 1 a 4 de 2026, el pronóstico solo superó el canal endémico en San Martín, con unos 170 casos por semana.`));
Dd.push(...tabla("oos", "Prueba fuera de muestra 2025 (XGBoost sin clima frente a persistencia) y MAE mediano en la prueba 2024", [461, 900, 900, 920, 920, COL - 4101],
  [[{ t: "", i: 0 }, { t: "2025", i: 1, span: 2, w: 1800, linea: true }, { t: "MAE mediano 2024", i: 3, span: 3, w: COL - 461 - 1800, linea: true }],
   [{ t: "*h*", i: 0 }, { t: "Deptos. mejores", i: 1 }, { t: "Mejora (%)", i: 2 }, { t: "XGBoost sin clima", i: 3 }, { t: "LSTM sin clima", i: 4 }, { t: "LSTM con clima", i: 5 }]],
  filasDe(leerTex("oos_lstm.tex"), -1)));
Dd.push(...figura("oos", "fuera_de_muestra_2025.png", COL, "Prueba fuera de muestra 2025 y pronóstico de 2026 (en naranja), a dos semanas, en las regiones de estudio."));

Dd.push(H1("Discusión"));
Dd.push(P(`A diferencia de lo reportado para Brasil {c:li2026}, el clima no mejoró de forma robusta el pronóstico semanal en el Perú una vez conocidos los casos recientes. Cuatro factores lo explican:`));
Dd.push(NUM("la resolución del clima, con celdas de unos 55 km promediadas por departamento;", "discusion"));
Dd.push(NUM("la fuerte inercia de los casos, que ya incorporan el efecto del clima de semanas previas;", "discusion"));
Dd.push(NUM("la escala departamental, que mezcla zonas con climas distintos;", "discusion"));
Dd.push(NUM(`que la influencia del clima sobre la transmisión opera con meses de rezago y de forma no lineal {c:lowe2021}.`, "discusion"));
Dd.push(P(`El caso de Lima sugiere, sin embargo, que el clima, y en especial El Niño costero, sirve como *contexto* de alerta temprana: indica cuándo una región sin historia puede volverse vulnerable.`));
Dd.push(P(`La lección metodológica principal es que **un solo corte temporal puede invertir las conclusiones**: la validación de 2022–2023 apoyaba H2 y la de nueve años la rechaza. Recomendamos la validación de origen móvil para estudios de este tipo.`));
Dd.push(PARRAFO_TITULADO("Limitaciones", ""));
Dd.push(VINETA("Los datos abiertos subestiman el total oficial."));
Dd.push(VINETA("Los modelos suponen que los casos recientes están completos, cuando en la práctica llegan con retraso de notificación."));
Dd.push(VINETA(`Faltan el punto de rocío y la presión, que sí usa {c:li2026}.`));
Dd.push(VINETA("La población de 2010–2017 es estimada."));
Dd.push(VINETA("El umbral departamental de brote es permisivo."));

Dd.push(H1("Conclusiones y trabajo futuro"));
Dd.push(P(`DengueSense pronostica el dengue semanal en 17 departamentos del Perú y supera a la persistencia en la mayoría de ellos, incluso en un año nunca visto (2025). Se confirmaron dos de las tres hipótesis:`));
Dd.push(VINETA("**H1, se confirma:** los factores climáticos difieren entre la costa y la selva."));
Dd.push(VINETA("**H3, se confirma:** el desempeño cae en periodos atípicos y sin historia."));
Dd.push(VINETA("**H2, no se sostiene:** con nueve años de prueba, el clima no mejora el pronóstico de forma consistente."));
Dd.push(P(`Anticipar el inicio de un brote sigue siendo el principal desafío.`));
Dd.push(P0("El trabajo futuro incluye:"));
Dd.push(VINETA(`usar clima de mayor resolución (ERA5-Land {c:hersbach2020} y estaciones del SENAMHI);`));
Dd.push(VINETA("trabajar a escala distrital;"));
Dd.push(VINETA("modelar el retraso de notificación;"));
Dd.push(VINETA("incorporar datos entomológicos y de movilidad;"));
Dd.push(VINETA("validar el LSTM con origen móvil."));
Dd.push(P(`El código, los cuadernos y una interfaz HTML autocontenida con mapas de riesgo están disponibles en el repositorio del proyecto: https://github.com/AdrianoJMG7/DengueSense.`));

Dd.push(H1("Referencias"));
// las referencias se agregan al final, cuando ya se conoce el orden de las citas

return { C1, A, W1, B, W2, Cc, W3, Dd };
}

// =============================================== documento ================
const numeracion = { config: [
  { reference: "vinetas", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                                      style: { paragraph: { indent: { left: 300, hanging: 200 } } } }] },
  ...["contribuciones", "discusion", "lista"].map((reference) => ({ reference, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] })),
] };

function seccion(children, columnas, primera = false) {
  return { properties: { ...(primera ? {} : { type: SectionType.CONTINUOUS }), page: PAGINA,
                         column: columnas === 2 ? { count: 2, space: SEP_COL, equalWidth: true } : { count: 1 } },
           children };
}

function construir() {
  construirContenido();                 // 1.a pasada: numera figuras y tablas
  ORDEN_CITAS.length = 0;
  const { C1, A, W1, B, W2, Cc, W3, Dd } = construirContenido();   // 2.a pasada: texto con numeros
  const referencias = ORDEN_CITAS.map((clave, i) => new Paragraph({ alignment: AlignmentType.JUSTIFIED,
    indent: { left: 360, hanging: 360 }, spacing: { after: 40 },
    children: [new TextRun({ text: `[${i + 1}]`, font: FUENTE, size: 16 }), new TextRun({ children: [new Tab()], font: FUENTE, size: 16 }),
               ...runs(REFS[clave], { size: 16 })] }));
  return new Document({
    creator: "DengueSense", title: "DengueSense: pronóstico semanal de dengue y alertas de brote en el Perú",
    description: "Artículo en formato IEEE", numbering: numeracion,
    styles: {
      default: { document: { run: { font: FUENTE, size: 20 }, paragraph: { spacing: { after: 0, line: 240 } } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FUENTE, size: 20, bold: false, color: "000000" },
          paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FUENTE, size: 20, italics: true, bold: false, color: "000000" },
          paragraph: { spacing: { before: 120, after: 60 }, keepNext: true, outlineLevel: 1 } },
      ],
    },
    sections: [
      seccion(C1, 1, true),
      seccion(A, 2),
      seccion(W1, 1),
      seccion(B, 2),
      seccion(W2, 1),
      seccion(Cc, 2),
      seccion(W3, 1),
      seccion([...Dd, ...referencias], 2),
    ],
  });
}

Packer.toBuffer(construir()).then((buffer) => {
  fs.writeFileSync(SALIDA, buffer);
  console.log(`Generado ${path.basename(SALIDA)}: ${(buffer.length / 1e6).toFixed(1)} MB, ${nFig} figuras, ${nTab} tablas, ${ORDEN_CITAS.length} referencias`);
});

// ─── NAVEGACIÓN ───────────────────────────────────────────────────────────────
function switchTab(sectionId, element) {
    document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active-view'));
    document.querySelectorAll('.menu a').forEach(a => a.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active-view');
    element.classList.add('active');
}

// ─── FORMATEO DE NÚMEROS ──────────────────────────────────────────────────────
function fmtME(val) {
    if (Math.abs(val) < 0.0001) return val.toExponential(3);
    return (val > 0 ? "+" : "") + val.toFixed(4);
}
function fmtSci(val) {
    if (val === undefined || val === null) return "-";
    if (Math.abs(val) < 0.0001) return val.toExponential(3);
    return val.toFixed(4);
}

// ─── MÉTRICAS + TABLA EFECTOS MARGINALES ─────────────────────────────────────
async function cargarMetricas() {
    try {
        const response = await fetch("/metricas");
        const data = await response.json();

        // KPIs
        document.getElementById("kpi-obs").innerText      = data.n_obs.toLocaleString();
        document.getElementById("kpi-tasa").innerText     = (data.tasa_informalidad * 100).toFixed(2) + "%";
        document.getElementById("kpi-r2").innerText       = data.pseudo_r2;
        document.getElementById("kpi-accuracy").innerText = (data.accuracy * 100).toFixed(1) + "%";

        // Pseudo R² en sección logit
        const r2Val = parseFloat(data.pseudo_r2);
        const r2El  = document.getElementById("pseudoR2Display");
        const r2Txt = document.getElementById("pseudoR2Texto");
        if (r2El)  r2El.innerText  = r2Val.toFixed(4);
        if (r2Txt) r2Txt.innerText = r2Val.toFixed(4);

        // Tabla de efectos marginales — 8 columnas completas
        const tbody = document.getElementById("tabla-efectos");
        if (tbody && data.efectos_marginales) {
            tbody.innerHTML = "";
            const maxAbs = Math.max(...data.efectos_marginales.map(e => Math.abs(e.efecto)));

            data.efectos_marginales.forEach(e => {
                const tr       = document.createElement("tr");
                const color    = e.efecto < 0 ? "#ff6f61" : "#00ffaa";
                const sigClass = e.significativa ? "badge-sig" : "badge-ns";
                const sigText  = e.significativa ? "p < 0.05"  : "n.s.";
                const pct      = ((Math.abs(e.efecto) / maxAbs) * 100).toFixed(0);
                const barColor = e.efecto < 0 ? "#ff6f61" : "#00ffaa";
                const pColor   = e.pvalor < 0.05 ? "#00ffaa" : "#8ea4d2";

                tr.innerHTML = `
                    <td><strong>${e.variable}</strong></td>
                    <td>
                        <div style="display:flex;align-items:center;gap:10px">
                            <span style="color:${color};font-family:monospace;min-width:90px">${fmtME(e.efecto)}</span>
                            <div style="flex:1;height:4px;background:rgba(255,255,255,.08);border-radius:2px;min-width:60px">
                                <div style="height:100%;width:${pct}%;background:${barColor};border-radius:2px"></div>
                            </div>
                        </div>
                    </td>
                    <td style="font-family:monospace;color:#8ea4d2">${fmtSci(e.std_err)}</td>
                    <td style="font-family:monospace">${e.z_stat !== undefined ? e.z_stat.toFixed(3) : "-"}</td>
                    <td style="font-family:monospace;color:${pColor}">${e.pvalor.toFixed(4)}</td>
                    <td style="font-family:monospace;color:#8ea4d2">${fmtSci(e.ci_low)}</td>
                    <td style="font-family:monospace;color:#8ea4d2">${fmtSci(e.ci_high)}</td>
                    <td><span class="badge ${sigClass}">${sigText}</span></td>
                `;
                tbody.appendChild(tr);
            });
        }

    } catch (err) {
        console.error("Error mapeando métricas:", err);
    }
}

// ─── GRÁFICAS ─────────────────────────────────────────────────────────────────
async function cargarGraficas() {
    try {
        const response = await fetch("/graficas");
        const data = await response.json();

        const mapeoImagenes = {
            "imgIngreso":  data.ingreso_mes,
            "imgHoras":    data.horas_semana,
            "imgEducacion":data.educacion,
            "imgEdad":     data.edad,
            "imgROC":      data.roc,
            "imgSector":   data.sector
        };

        Object.keys(mapeoImagenes).forEach(id => {
            const imgElement = document.getElementById(id);
            if (imgElement && mapeoImagenes[id]) {
                imgElement.src = "data:image/png;base64," + mapeoImagenes[id];
                imgElement.onload = function () {
                    imgElement.classList.add("loaded");
                    imgElement.parentElement.classList.remove("skeleton");
                };
            }
        });
    } catch (err) {
        console.error("Error cargando gráficos:", err);
    }
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    cargarMetricas();
    cargarGraficas();
});
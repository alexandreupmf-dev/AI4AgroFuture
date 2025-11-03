// static/app.js (avisa quando não veio nada)
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("btn-refresh");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Atualizando...";
    try {
      const resp = await fetch("/api/refresh_sinais", { method: "POST" });
      const data = await resp.json();
      if (data && typeof data.qtd === "number" && data.qtd === 0) {
        alert("Não foi possível coletar novos sinais agora. Verifique sua internet ou tente novamente em alguns minutos.");
      }
      window.location.reload();
    } catch (e) {
      console.error(e);
      alert("Falha ao atualizar sinais.");
    } finally {
      btn.disabled = false;
      btn.textContent = old;
    }
  });
});


// StockSense AI — Custom JS enhancements

// ── Clock ──────────────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById('header-clock');
  if (!el) return;
  const now = new Date();
  const opts = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
  const time = now.toLocaleTimeString('en-US', opts);
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
  el.textContent = `${time} ${tz}`;
}
setInterval(updateClock, 1000);
updateClock();

// ── Price Flash ────────────────────────────────────────────────
let lastPrice = null;
function flashPrice(newPrice) {
  const el = document.getElementById('price-display');
  if (!el || newPrice === lastPrice) return;
  const cls = newPrice > lastPrice ? 'flash-up' : 'flash-down';
  el.classList.remove('flash-up', 'flash-down');
  void el.offsetWidth; // reflow
  el.classList.add(cls);
  lastPrice = newPrice;
  setTimeout(() => el.classList.remove(cls), 800);
}

// ── Animated Number Counter ────────────────────────────────────
function animateNumber(element, start, end, duration, prefix, suffix, decimals) {
  const startTime = performance.now();
  const step = (currentTime) => {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    const current = start + (end - start) * eased;
    element.textContent = `${prefix || ''}${current.toFixed(decimals || 2)}${suffix || ''}`;
    if (progress < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ── Keyboard Shortcuts ─────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  // S: focus stock search
  if (e.key === 's' || e.key === 'S') {
    const input = document.querySelector('.stock-search input');
    if (input) { input.focus(); e.preventDefault(); }
  }

  // F: focus forecast tab
  if (e.key === 'f' || e.key === 'F') {
    const tabs = document.querySelectorAll('.main-tabs .nav-link');
    if (tabs[1]) tabs[1].click();
  }

  // T: focus technical tab
  if (e.key === 't' || e.key === 'T') {
    const tabs = document.querySelectorAll('.main-tabs .nav-link');
    if (tabs[2]) tabs[2].click();
  }

  // Escape: blur all inputs
  if (e.key === 'Escape') {
    document.activeElement.blur();
  }
});

// ── Smooth Tab Transitions ─────────────────────────────────────
const observer = new MutationObserver((mutations) => {
  mutations.forEach((m) => {
    m.addedNodes.forEach((node) => {
      if (node.nodeType === 1 && node.classList.contains('tab-content-panel')) {
        node.style.opacity = '0';
        node.style.transform = 'translateY(8px)';
        requestAnimationFrame(() => {
          node.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
          node.style.opacity = '1';
          node.style.transform = 'translateY(0)';
        });
      }
    });
  });
});

observer.observe(document.body, { childList: true, subtree: true });

// ── Tooltip Titles ─────────────────────────────────────────────
function initTooltips() {
  document.querySelectorAll('[data-tooltip]').forEach((el) => {
    el.title = el.dataset.tooltip;
  });
}

// ── Copy to Clipboard ──────────────────────────────────────────
window.copyToClipboard = function(text) {
  navigator.clipboard.writeText(text).then(() => {
    const toast = document.createElement('div');
    toast.textContent = 'Copied!';
    toast.style.cssText = `
      position: fixed; bottom: 20px; right: 20px;
      background: rgba(0,200,255,0.15); border: 1px solid rgba(0,200,255,0.3);
      color: #00c8ff; padding: 8px 16px; border-radius: 8px;
      font-size: 13px; font-weight: 600; z-index: 9999;
      animation: fadeSlideUp 0.3s ease;
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2000);
  });
};

// ── Chart Resize on Sidebar Toggle ────────────────────────────
window.addEventListener('resize', () => {
  const plots = document.querySelectorAll('.js-plotly-plot');
  if (window.Plotly) {
    plots.forEach((p) => window.Plotly.Plots.resize(p));
  }
});

// ── Lazy init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTooltips();
  updateClock();
});

// Re-init after Dash re-renders
if (window._dash_callbacks) {
  const orig = window._dash_callbacks;
  window._dash_callbacks = new Proxy(orig, {
    set(target, prop, value) {
      target[prop] = value;
      initTooltips();
      return true;
    }
  });
}

"use strict";

// ---------- UI多言語化の基盤 ----------
// locale辞書は locales/<locale>.js が I18N.locales に登録する (ja が正)。
// ビルドなし・素の<script>で動かすため、app.js より先に読み込む。
const I18N = {
  locales: {},
  current: "ja",
  STORAGE_KEY: "ssLang",

  // 選択肢の表示名は各言語そのものの表記 (翻訳しない)
  labels: { "ja": "日本語", "en": "English", "zh-CN": "中文 (简体)" },

  supported() { return Object.keys(this.locales); },

  normalize(tag) {
    const value = String(tag || "").toLowerCase();
    if (value === "ja" || value.startsWith("ja-")) return "ja";
    if (value.startsWith("zh")) return "zh-CN";
    return "en";
  },

  detect() {
    try {
      const saved = localStorage.getItem(this.STORAGE_KEY);
      if (saved && this.locales[saved]) return saved;
    } catch (_) { /* ストレージ禁止時は言語設定も保存できない */ }
    const langs = navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language || "ja"];
    for (const tag of langs) {
      const locale = this.normalize(tag);
      if (this.locales[locale]) return locale;
    }
    return "ja";
  },

  init() {
    this.current = this.detect();
    this.syncLangAttr();
    this.applyStatic();
  },

  setLocale(locale) {
    if (!this.locales[locale] || locale === this.current) return;
    this.current = locale;
    try { localStorage.setItem(this.STORAGE_KEY, locale); } catch (_) {}
    this.syncLangAttr();
    this.applyStatic();
    document.dispatchEvent(new CustomEvent("i18nchange"));
  },

  syncLangAttr() {
    document.documentElement.lang = this.current;
  },

  // 未登録キーは ja → キー本体の順にフォールバックし、空白を返さない
  t(key, params) {
    const dict = this.locales[this.current];
    const value = (dict && dict[key]) !== undefined ? dict[key] : this.locales.ja[key];
    if (value === undefined) return key;
    if (!params) return value;
    return value.replace(/\{([a-zA-Z_][a-zA-Z0-9_]*)\}/g, (m, name) =>
      params && Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : m);
  },

  has(key) {
    const dict = this.locales[this.current];
    return (dict && dict[key] !== undefined) || this.locales.ja[key] !== undefined;
  },

  // 静的HTMLの翻訳: [data-i18n]=textContent / [data-i18n-html]=innerHTML (静的マークアップ込み) /
  // [data-i18n-attr]="title:key placeholder:key" (属性)
  applyStatic() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.dataset.i18n;
      if (this.has(key)) el.textContent = this.t(key);
    });
    document.querySelectorAll("[data-i18n-html]").forEach((el) => {
      const key = el.dataset.i18nHtml;
      if (this.has(key)) el.innerHTML = this.t(key);
    });
    document.querySelectorAll("[data-i18n-attr]").forEach((el) => {
      el.dataset.i18nAttr.split(/\s+/).forEach((pair) => {
        const [attr, key] = pair.split(":");
        if (attr && key && this.has(key)) el.setAttribute(attr, this.t(key));
      });
    });
  },
};
window.I18N = I18N;

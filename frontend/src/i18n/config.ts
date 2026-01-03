import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import zhTranslations from './locales/zh.json';
import enTranslations from './locales/en.json';

// Read saved language setting from localStorage, default to English
const savedLanguage = localStorage.getItem('language') || 'en';

i18n
  .use(initReactI18next)
  .init({
    resources: {
      zh: {
        translation: zhTranslations
      },
      en: {
        translation: enTranslations
      }
    },
    lng: savedLanguage, // Default language
    fallbackLng: 'en', // Fallback language
    interpolation: {
      escapeValue: false // React already escapes, no need for i18n to escape again
    }
  });

export default i18n;

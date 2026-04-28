import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        coral: {
          50: "#fff5f1",
          100: "#ffe6dc",
          300: "#ffb89c",
          400: "#ff9b7a",
          500: "#ff6b4a",
          600: "#ee5532",
          700: "#c84327",
        },
      },
      backgroundImage: {
        "coral-grad": "linear-gradient(135deg, #ff9b7a 0%, #ff6b4a 100%)",
      },
      boxShadow: {
        glass: "0 8px 32px 0 rgba(31, 38, 135, 0.06)",
      },
    },
  },
  plugins: [],
} satisfies Config;

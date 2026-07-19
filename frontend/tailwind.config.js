/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Verdict colours. `uncertain` is a first-class slate, NOT a muted grey
        // footnote (T63) -- it must read as an equal third state.
        real: "#2a9d8f",
        fake: "#e76f51",
        uncertain: "#64748b",
      },
    },
  },
  plugins: [],
};

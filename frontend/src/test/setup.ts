import "@testing-library/jest-dom/vitest";

// jsdom does not implement these, and components touch them. Stub minimally.
// globalThis (not `global`) so this type-checks without @types/node.
if (!globalThis.URL.createObjectURL) {
  globalThis.URL.createObjectURL = () => "blob:mock";
  globalThis.URL.revokeObjectURL = () => {};
}

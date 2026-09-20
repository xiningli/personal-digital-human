// Tailwind v3 (downgraded from v4, ADR: see next.config.ts note). Classic plugin form: no
// native binary in the chain, so this survives a clean Next.js dev/build rebuild without the
// lightningcss module-resolution failure v4's @tailwindcss/postcss carried.
const config = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};

export default config;

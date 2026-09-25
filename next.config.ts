import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Dev-only: hosts allowed to load /_next resources besides localhost. Add your own
  // LAN/Tailscale hostnames here when serving the dev server to other machines.
  allowedDevOrigins: ["127.0.0.1"],
  // Standalone project with sibling apps (each with their own lockfile) one level up under
  // ~/digital-human; pin the root so Turbopack doesn't infer that shared parent as the
  // workspace root and resolve node_modules against it instead of this project's own.
  turbopack: { root: __dirname },
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;

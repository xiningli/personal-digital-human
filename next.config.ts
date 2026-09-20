import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Dev-only: hosts allowed to load /_next resources besides localhost (same list as the voice studio).
  allowedDevOrigins: ["127.0.0.1", "noname-tuf", "noname-tuf.local", "192.168.8.196", "100.126.201.107", "*.ts.net"],
  // Standalone project with sibling apps (each with their own lockfile) one level up under
  // ~/digital-human; pin the root so Turbopack doesn't infer that shared parent as the
  // workspace root and resolve node_modules against it instead of this project's own.
  turbopack: { root: __dirname },
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;

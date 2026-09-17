import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Dev-only: hosts allowed to load /_next resources besides localhost (same list as the voice studio).
  allowedDevOrigins: ["127.0.0.1", "noname-tuf", "noname-tuf.local", "192.168.8.196", "100.126.201.107", "*.ts.net"],
};

export default nextConfig;

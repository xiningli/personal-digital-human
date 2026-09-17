import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "Personal Digital Human",
  description: "Preference learning for the digital human's body language",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex">
        <Sidebar />
        <main className="flex-1 overflow-auto"><div className="max-w-5xl mx-auto px-8 py-8">{children}</div></main>
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vision Inference",
  description: "Upload image or video media for object detection.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

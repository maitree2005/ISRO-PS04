"use client";

import dynamic from "next/dynamic";

const MapComponent = dynamic(() => import("../components/Map"), {
  ssr: false,
  loading: () => <div style={{ padding: 20 }}>Loading map...</div>,
});

export default function Home() {
  return <MapComponent />;
}
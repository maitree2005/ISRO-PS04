"use client"
import dynamic from 'next/dynamic'
import React from 'react'

const Map = dynamic(() => import('../components/Map'), { ssr: false })

export default function Page() {
  return (
    <main style={{ height: '100vh', margin: 0 }}>
      <Map />
    </main>
  )
}

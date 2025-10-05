import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    allowedHosts: [
      'shinkaevolve.preview.emergentagent.com',
      'bf7227fc-ba83-4dd5-b96c-522be2796f63.preview.emergentagent.com',
      '.emergentagent.com',
      'localhost'
    ]
  },
  define: {
    'process.env': {}
  }
})

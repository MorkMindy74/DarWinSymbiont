import React, { useEffect } from 'react';
import { AlertCircle } from 'lucide-react';

/**
 * Component to redirect from preview to localhost with instructions
 */
function PreviewRedirect() {
  const isPreview = window.location.hostname.includes('preview.emergentagent.com');

  useEffect(() => {
    // Auto-redirect after 5 seconds if on preview
    if (isPreview) {
      const timer = setTimeout(() => {
        // Try to redirect to localhost
        window.location.href = 'http://localhost:3000';
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [isPreview]);

  if (!isPreview) {
    return null; // Don't show anything if already on localhost
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl p-8 max-w-2xl mx-4">
        <div className="flex items-center space-x-3 mb-6">
          <AlertCircle className="w-8 h-8 text-orange-500" />
          <h2 className="text-2xl font-bold text-gray-900">
            Accesso da Preview Esterna Rilevato
          </h2>
        </div>

        <div className="space-y-4 text-gray-700">
          <p className="text-lg">
            Stai accedendo dalla preview esterna che richiede agent attivo.
          </p>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="font-semibold text-blue-900 mb-2">
              ✅ Soluzione Raccomandata: Usa Localhost
            </p>
            <ol className="list-decimal list-inside space-y-2 text-blue-800">
              <li>Clicca sulla barra degli indirizzi del browser (in alto)</li>
              <li>Cancella l'URL corrente</li>
              <li>Digita: <code className="bg-blue-100 px-2 py-1 rounded">http://localhost:3000</code></li>
              <li>Premi Invio</li>
            </ol>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <p className="font-semibold text-green-900 mb-2">
              🚀 Auto-redirect tra 5 secondi...
            </p>
            <p className="text-green-800">
              Ti reindirizzeremo automaticamente a localhost.
            </p>
          </div>

          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <p className="font-semibold text-gray-900 mb-2">
              ℹ️ Perché Localhost?
            </p>
            <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
              <li>Connessione diretta al server (nessun proxy)</li>
              <li>WebSocket funziona sempre</li>
              <li>Nessuna dipendenza da infrastruttura esterna</li>
              <li>Migliore per development e testing</li>
            </ul>
          </div>
        </div>

        <div className="mt-6 flex space-x-3">
          <button
            onClick={() => window.location.href = 'http://localhost:3000'}
            className="btn-primary flex-1"
          >
            Vai a Localhost Ora
          </button>
        </div>
      </div>
    </div>
  );
}

export default PreviewRedirect;

import React from 'react';
import { AlertCircle, X } from 'lucide-react';

/**
 * Component to show warning about preview limitations (no redirect)
 */
function PreviewRedirect() {
  const [dismissed, setDismissed] = React.useState(false);
  const isPreview = window.location.hostname.includes('preview.emergentagent.com');

  if (!isPreview || dismissed) {
    return null; // Don't show if not on preview or dismissed
  }

  return (
    <div className="fixed top-4 right-4 bg-yellow-50 border-2 border-yellow-400 rounded-lg shadow-xl p-4 max-w-md z-50">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0" />
          <h3 className="font-semibold text-yellow-900">
            Preview Esterna - Limitazioni
          </h3>
        </div>
        <button
          onClick={() => setDismissed(true)}
          className="text-yellow-600 hover:text-yellow-800"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="text-sm text-yellow-800 space-y-2">
        <p>
          ⚠️ La preview esterna potrebbe avere problemi WebSocket.
        </p>
        <p className="font-medium">
          💡 Se l'evolution non si aggiorna, riavvia l'agent da app.emergent.sh
        </p>
      </div>
    </div>
  );
}

export default PreviewRedirect;

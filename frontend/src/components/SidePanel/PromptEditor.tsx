import { useEffect, useRef, useState } from 'react';

interface Prompts { user_prompt: string; system_prompt: string }
interface PromptState { primary: Prompts | null; confirm: Prompts | null }

async function fetchPrompts(): Promise<PromptState> {
  const r = await fetch('/api/vlm/prompt');
  return r.json();
}

async function pushPrompt(stage: string, user_prompt: string, system_prompt: string) {
  await fetch('/api/vlm/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stage, user_prompt, system_prompt }),
  });
}

export function PromptEditor() {
  const [open, setOpen] = useState(false);
  const [stage, setStage] = useState<'primary' | 'confirm'>('primary');
  const [prompts, setPrompts] = useState<PromptState>({ primary: null, confirm: null });
  const [userText, setUserText] = useState('');
  const [sysText, setSysText] = useState('');
  const [status, setStatus] = useState('');
  const loadedRef = useRef(false);

  useEffect(() => {
    if (open && !loadedRef.current) {
      loadedRef.current = true;
      fetchPrompts().then((p) => {
        setPrompts(p);
        const active = stage === 'primary' ? p.primary : p.confirm;
        if (active) { setUserText(active.user_prompt); setSysText(active.system_prompt); }
      });
    }
  }, [open]);

  useEffect(() => {
    const active = stage === 'primary' ? prompts.primary : prompts.confirm;
    if (active) { setUserText(active.user_prompt); setSysText(active.system_prompt); }
  }, [stage, prompts]);

  const reload = () => {
    loadedRef.current = false;
    fetchPrompts().then((p) => {
      setPrompts(p);
      loadedRef.current = true;
    });
  };

  const apply = async () => {
    setStatus('Applying…');
    try {
      await pushPrompt(stage, userText, sysText);
      setStatus('Applied');
    } catch {
      setStatus('Error');
    }
    setTimeout(() => setStatus(''), 2000);
  };

  const active = stage === 'primary' ? prompts.primary : prompts.confirm;

  return (
    <div className="card prompt-editor-card">
      <h2 style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => setOpen((o) => !o)}>
        VLM Prompt {open ? '▲' : '▼'}
      </h2>
      {open && (
        <div className="prompt-editor-body">
          <div className="prompt-stage-row">
            <select value={stage} onChange={(e) => setStage(e.target.value as 'primary' | 'confirm')}>
              <option value="primary">Phase 1 — primary</option>
              {prompts.confirm && <option value="confirm">Phase 2 — confirm</option>}
            </select>
            <button className="prompt-btn" onClick={reload} title="Reload from server">↺</button>
          </div>
          {active === null
            ? <p className="prompt-unavail">Stage not available (no two-stage pipeline)</p>
            : <>
                <label className="prompt-label">System prompt</label>
                <textarea className="prompt-ta" rows={4} value={sysText} onChange={(e) => setSysText(e.target.value)} />
                <label className="prompt-label">User prompt</label>
                <textarea className="prompt-ta" rows={12} value={userText} onChange={(e) => setUserText(e.target.value)} />
                <div className="prompt-actions">
                  <button className="prompt-btn apply" onClick={apply}>Apply</button>
                  {status && <span className="prompt-status">{status}</span>}
                </div>
              </>
          }
        </div>
      )}
    </div>
  );
}

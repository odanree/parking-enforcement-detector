import { StatsCard }     from './StatsCard';
import { VlmQueue }      from './VlmQueue';
import { EventLog }      from './EventLog';
import { PromptEditor }  from './PromptEditor';

export function SidePanel() {
  return (
    <aside className="side-panel">
      <StatsCard />
      <VlmQueue />
      <PromptEditor />
      <EventLog />
    </aside>
  );
}

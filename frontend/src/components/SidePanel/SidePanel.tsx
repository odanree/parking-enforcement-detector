import { StatsCard } from './StatsCard';
import { VlmQueue }  from './VlmQueue';
import { EventLog }  from './EventLog';

export function SidePanel() {
  return (
    <aside className="side-panel">
      <StatsCard />
      <VlmQueue />
      <EventLog />
    </aside>
  );
}

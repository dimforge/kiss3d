import {useEffect, useState, useMemo, useRef} from 'react';
import Layout from '@theme/Layout';
import styles from './examples.module.css';

// Example categories and files
const examples = {
  '2D': [
    { name: 'window', title: 'Window', description: 'Basic window creation' },
    { name: 'rectangle', title: 'Rectangle', description: '2D rectangle' },
    { name: 'primitives2d', title: 'Primitives 2D', description: '2D shapes rendering' },
    { name: 'lines2d', title: 'Lines 2D', description: '2D line rendering' },
    { name: 'points2d', title: 'Points 2D', description: '2D point rendering' },
    { name: 'polylines2d', title: 'Polylines 2D', description: '2D polylines' },
    { name: 'instancing2d', title: 'Instancing 2D', description: '2D instancing' },
    { name: 'mouse_events', title: 'Mouse Events', description: 'Mouse interaction' },
  ],
  '3D': [
    { name: 'ui', title: 'UI', description: 'Simple widgets' },
    { name: 'cube', title: 'Cube', description: 'Simple rotating cube' },
    { name: 'primitives', title: 'Primitives 3D', description: 'All 3D primitive shapes' },
    { name: 'primitives_scale', title: 'Scaled Primitives', description: 'Primitives with scaling' },
    { name: 'quad', title: 'Quad', description: 'Quad rendering' },
    { name: 'lines', title: 'Lines 3D', description: '3D line rendering' },
    { name: 'points', title: 'Points 3D', description: 'Point cloud rendering' },
    { name: 'polylines', title: 'Polylines', description: 'Polyline physics simulation' },
    { name: 'polyline_strip', title: 'Polyline Strip', description: 'Connected polylines' },
    { name: 'wireframe', title: 'Wireframe', description: 'Wireframe rendering' },
    { name: 'multi_light', title: 'Multiple Lights', description: 'Multiple light sources' },
    { name: 'custom_mesh', title: 'Custom Mesh', description: 'Custom geometry' },
    { name: 'custom_mesh_shared', title: 'Shared Mesh', description: 'Shared mesh instances' },
    { name: 'procedural', title: 'Procedural', description: 'Procedural mesh generation' },
    { name: 'group', title: 'Groups', description: 'Hierarchical transforms' },
    { name: 'post_processing', title: 'Post Processing', description: 'Visual effects' },
    { name: 'instancing3d', title: 'Instancing 3D', description: '3D instancing' },
    { name: 'custom_material', title: 'Custom Material', description: 'Custom shaders' },
  ]
};

type Example = {
  name: string;
  title: string;
  description: string;
};

export default function Examples(): JSX.Element {
  const [filter, setFilter] = useState('');
  const [selected, setSelected] = useState<string | null>(null);
  const [activeDemo, setActiveDemo] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  // Handle URL hash for deep linking
  useEffect(() => {
    const hash = window.location.hash.slice(1);
    if (hash) {
      setSelected(hash);
    } else {
      // Default to cube
      setSelected('ui');
    }

    const handleHashChange = () => {
      const newHash = window.location.hash.slice(1);
      if (newHash) setSelected(newHash);
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Handle demo transitions - clear iframe first to release WebGPU context
  useEffect(() => {
    if (selected === activeDemo) return;

    // Clear the current demo first
    setIsLoading(true);

    // Force iframe cleanup by setting src to blank first
    if (iframeRef.current) {
      iframeRef.current.src = 'about:blank';
    }
    setActiveDemo(null);

    // Wait for the iframe to be cleared and GPU context to be released
    // 500ms gives browsers more time to garbage collect WebGPU resources
    const timer = setTimeout(() => {
      setActiveDemo(selected);
      setIsLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, [selected]);

  // Filter examples
  const filteredExamples = useMemo(() => {
    if (!filter.trim()) return examples;

    const searchTerms = filter.toLowerCase().split(/\s+/);
    const filtered: typeof examples = {};

    for (const [category, items] of Object.entries(examples)) {
      const matchingItems = items.filter((item) => {
        const searchText = `${item.name} ${item.title} ${item.description} ${category}`.toLowerCase();
        return searchTerms.every((term) => searchText.includes(term));
      });
      if (matchingItems.length > 0) {
        filtered[category] = matchingItems;
      }
    }

    return filtered;
  }, [filter]);

  const handleSelect = (name: string) => {
    setSelected(name);
    window.location.hash = name;
    // Close panel on mobile after selection
    if (window.innerWidth < 768) {
      setPanelOpen(false);
    }
  };

  const totalExamples = Object.values(examples).flat().length;
  const filteredCount = Object.values(filteredExamples).flat().length;

  return (
    <Layout
      title="Examples"
      description="Interactive kiss3d examples running in your browser"
      noFooter
    >
      <div className={styles.container}>
        {/* Panel Toggle Button */}
        <button
          className={`${styles.panelToggle} ${panelOpen ? styles.panelToggleOpen : ''}`}
          onClick={() => setPanelOpen(!panelOpen)}
          aria-label="Toggle panel"
        >
          {panelOpen ? '◀' : '▶'}
        </button>

        {/* Left Panel */}
        <div className={`${styles.panel} ${panelOpen ? styles.panelOpen : ''}`}>
          <div className={styles.search}>
            <input
              type="text"
              placeholder="Search examples..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className={styles.searchInput}
            />
            {filter && (
              <button
                className={styles.clearSearch}
                onClick={() => setFilter('')}
                aria-label="Clear search"
              >
                ×
              </button>
            )}
          </div>

          <div className={styles.count}>
            {filter ? `${filteredCount} of ${totalExamples}` : `${totalExamples} examples`}
          </div>

          <div className={styles.categories}>
            {Object.entries(filteredExamples).map(([category, items]) => (
              <div key={category} className={styles.category}>
                <h3 className={styles.categoryTitle}>{category}</h3>
                <div className={styles.cards}>
                  {items.map((example) => (
                    <button
                      key={example.name}
                      className={`${styles.card} ${selected === example.name ? styles.cardSelected : ''}`}
                      onClick={() => handleSelect(example.name)}
                    >
                      {example.title}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Scrim for mobile */}
        {panelOpen && (
          <div
            className={styles.scrim}
            onClick={() => setPanelOpen(false)}
          />
        )}

        {/* Main Viewer */}
        <div className={styles.viewer}>
          {activeDemo ? (
            <>
              <iframe
                ref={iframeRef}
                key={activeDemo}
                src={`/demos/${activeDemo}/`}
                title={activeDemo}
                className={styles.viewerFrame}
              />
              <div className={styles.viewerControls}>
                <a
                  href={`https://github.com/sebcrozet/kiss3d/blob/master/examples/${selected}.rs`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={styles.sourceLink}
                >
                  &lt;/&gt; Source
                </a>
              </div>
            </>
          ) : isLoading ? (
            <div className={styles.placeholder}>
              Loading...
            </div>
          ) : (
            <div className={styles.placeholder}>
              Select an example from the panel
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}

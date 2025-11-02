# Story-Specific Folder Organization

## Overview

Each generated story now gets its own unique folder to prevent overwriting previous stories. You can build a library of generated stories!

## Folder Structure

```
story_output/
├── index.json                                    # Master index of all stories
├── Pride_and_Prejudice_Scifi_2025-11-02_14-30/  # Story-specific folder
│   ├── metadata.json                             # Story metadata
│   ├── complete_story.txt                        # Full readable story
│   └── chapters/                                 # Individual chapter JSONs
│       ├── 01_Opening_Image.json
│       ├── 02_Theme_Stated.json
│       └── ...
├── Sherlock_Holmes_Medieval_2025-11-02_15-45/   # Another story
│   ├── metadata.json
│   ├── complete_story.txt
│   └── chapters/
└── ...
```

## Folder Naming Convention

Folders are automatically named using this pattern:
```
{Source_Title}_{Setting}_{Timestamp}
```

**Examples:**
- `Pride_and_Prejudice_Scifi_2025-11-02_14-30`
- `Star_Wars_Medieval_2025-11-02_16-00`
- `Sherlock_Holmes_Space_2025-11-02_17-15`

**Details:**
- **Source Title**: Cleaned to remove special characters, max 30 characters
- **Setting**: The genre/setting chosen (Scifi, Medieval, Space, etc.)
- **Timestamp**: `YYYY-MM-DD_HH-MM` format to ensure uniqueness

## Story Index (`story_output/index.json`)

The index file tracks all generated stories with metadata:

```json
{
  "stories": [
    {
      "folder": "Pride_and_Prejudice_Scifi_2025-11-02_14-30",
      "title": "Pride and Prejudice",
      "setting": "sci-fi",
      "hero_arc": "Gets what they want and what they need",
      "created_at": "2025-11-02T14:30:00",
      "status": "complete",
      "completed_at": "2025-11-02T15:45:00"
    },
    {
      "folder": "Sherlock_Holmes_Medieval_2025-11-02_16-00",
      "title": "Sherlock Holmes",
      "setting": "medieval",
      "hero_arc": "Gets what they want but not what they need",
      "created_at": "2025-11-02T16:00:00",
      "status": "in_progress"
    }
  ]
}
```

**Status Values:**
- `in_progress`: Story generation started but not finished
- `complete`: Story fully generated and compiled

## Benefits

1. **No Overwriting**: Each story generation creates a new folder
2. **Story Library**: Build a collection of different story variations
3. **Easy Comparison**: Compare different settings/hero arcs for same source
4. **Organization**: All files for one story in one place
5. **Tracking**: Index file shows all stories at a glance

## Usage Examples

### Browse all generated stories
```bash
cat story_output/index.json
```

### Read a specific story
```bash
cat story_output/Pride_and_Prejudice_Scifi_2025-11-02_14-30/complete_story.txt
```

### View story metadata
```bash
cat story_output/Pride_and_Prejudice_Scifi_2025-11-02_14-30/metadata.json
```

### List all story folders
```bash
ls -la story_output/
```

### Compare different versions
```bash
# Generate Pride and Prejudice in Sci-Fi
# Generate Pride and Prejudice in Medieval
# Both versions preserved in separate folders!
diff story_output/Pride_and_Prejudice_Scifi_*/complete_story.txt \
     story_output/Pride_and_Prejudice_Medieval_*/complete_story.txt
```

## Implementation Details

### When is the folder created?

The story-specific folder is created in **Agent 3: Setting Selection** after the user chooses the setting. This is the earliest point where we have both:
1. Source story title
2. Setting/genre

### What gets saved to the folder?

- **metadata.json**: Story configuration and statistics
- **complete_story.txt**: Full readable story text
- **chapters/*.json**: Individual chapter data with quality metrics

### Index updates

The index is updated at two points:
1. **Story start** (Setting Agent): Adds entry with `status: "in_progress"`
2. **Story completion** (Save Metadata): Updates to `status: "complete"` with completion timestamp

## Migration Note

If you have existing stories in the old `story_output/` structure (before this feature), they won't be in the index. They won't be overwritten - the system will create new story-specific folders alongside them.

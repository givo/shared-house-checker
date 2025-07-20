# Blueprint Analysis & Automated Visual Checklist

## Project Overview

This project automates visual checklist tasks that are currently performed manually by humans when reviewing building floor plans. Uses traditional image processing techniques in Python to analyze AutoCAD blueprints and validate compliance requirements.

## Input

**Blueprint image**: AutoCAD-generated floor plan (PDF or image format)

**Features**: 
  - Colored apartment units (each apartment has unique color)
  - Black arrows pointing to walls indicating entrances
  - Black circles with apartment numbers
  - Storage room entrances (optional secondary arrows)

## Current Feature: Entrance Detection

Validates that each apartment has a proper entrance - replacing manual visual inspection.

### Detection Logic

- **2+ arrows per apartment**: Main entrance exists
- **1 arrow**: Check if points to storage room or main area
- **0 arrows**: No entrance detected

## Output

Dictionary mapping each apartment to its validation status:
```python
{
    "apartment_1": "has_entrance",
    "apartment_2": "no_entrance", 
    "apartment_3": "uncertain"
}
```

## Method

Traditional computer vision pipeline: color segmentation → feature detection → spatial association → rule validation

## Future Extensions

This framework can be extended to automate other visual checklist items like room size validation, accessibility compliance, fire safety requirements, etc.
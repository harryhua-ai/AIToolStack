# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Auto-sync Class Names list with Num Classes field
- Frontend validation: Num Classes must be >= 1
- Improved error handling with proper object-to-string conversion

### Fixed
- Error handling showing "[object Object]" instead of actual error messages (5 locations)
- Calibration images count showing "{{count}}" placeholder instead of actual count
- Num Classes validation error (Pydantic: value must be >= 1)
- Model Space internationalization issues (English/Chinese display)

### Changed
- Standardized i18n parameter passing pattern
- Improved error message consistency across all API calls

## [2026-03-05]

### Fixed
- Initial Model Space UI fixes
- Calibration upload functionality
- I18N display issues

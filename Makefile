# Variables
IDX_FILE := docs/index.md
APP_FILE := docs/app.md
LIBNAME  := seroepi

# Phony targets aren't real files
.PHONY: all docs clean build dev sync test

# Default target when you just run `make`
all: docs

build:
	@echo "🏗️ Building $(LIBNAME) package for distribution using uv..."
	uv build
	@echo "✅ Build complete! Artifacts are in the dist/ directory."

dev:
	@echo "🚧 Installing $(LIBNAME) and dev dependencies..."
	uv sync --all-groups
	@echo "✅ Development installation complete!"

sync:
	@echo "💫 Syncing environment dependencies..."
	uv sync
	@echo "✅ Environment synced!"

#test:
#	@echo "🧪 Running pytest..."
#	uv run --group test pytest -v
#	@echo "✅ Tests complete!"

# The master docs build target
docs: $(APP_FILE) | $(IDX_FILE)
	@echo "🚀 Building static site..."
	uv run --group docs zensical build --clean
	@echo "✅ Documentation built successfully!"

# Generate the CLI markdown
$(APP_FILE): src/seroepi/app/README.md
	@echo "📂 Copying app README to ${@}..."
	cp src/seroepi/app/README.md $@
	@echo "✅ Completed successfully!"

$(IDX_FILE): README.md
	@echo "📂 Copying README to ${@}..."
	cp README.md $@
	@echo "✅ Completed successfully!"

# Clean target to wipe generated docs and Python build artifacts
clean:
	@echo "🧹 Cleaning up generated documentation and build artifacts..."
	rm -rf $(APP_FILE) dist/ build/ *.egg-info .pytest_cache
	@echo "✅ Clean complete."
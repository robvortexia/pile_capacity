import pkg_resources
import os

def get_package_sizes():
    packages = {}
    for package in pkg_resources.working_set:
        try:
            path = os.path.dirname(package.location)
            size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(package.location)
                for filename in filenames
            )
            packages[package.key] = size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            print(f"Error processing {package.key}: {e}")
    return packages

# Print package sizes
for package, size in sorted(get_package_sizes().items(), key=lambda x: x[1], reverse=True):
    print(f"{package}: {size:.2f} MB")
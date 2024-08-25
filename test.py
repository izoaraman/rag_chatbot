import pkg_resources

with open('requirements.txt', 'r') as f:
    required_packages = f.read().splitlines()  # Read package names from file

for package_name in required_packages:
    try:
        package = pkg_resources.get_distribution(package_name)
        print(f"{package.project_name}=={package.version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package_name} is not installed.")
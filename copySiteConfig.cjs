const fs = require('fs');
const path = require('path');

const inputFilePath = path.join(__dirname, 'src/siteConfig.json');
const outputDirectory = path.join(__dirname, 'public');
const outputFilePath = path.join(outputDirectory, 'siteConfig.json');

if (!fs.existsSync(outputDirectory)) {
  fs.mkdirSync(outputDirectory, { recursive: true });
}

// Read the existing siteConfig.json file
const siteConfigData = JSON.parse(fs.readFileSync(inputFilePath, 'utf8'));

// Write the siteConfig.json data to a new file in the public folder
fs.writeFileSync(outputFilePath, JSON.stringify(siteConfigData, null, 2), 'utf8');

console.log('siteConfig.json copied to public folder');

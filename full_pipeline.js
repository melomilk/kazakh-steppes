// ============================================================
// AIST 2026 Research — Betpak-Dala / Zhanaarka Steppe Degradation
// Google Earth Engine code (code.earthengine.google.com)
// ============================================================

// ------------------------------------------------------------
// STEP 1: Basic NDVI visualization (exploration phase)
// ------------------------------------------------------------
var region = ee.Geometry.Rectangle([67, 49, 75, 54]);

var image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filterDate('2023-06-01', '2023-08-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

var ndvi = image.normalizedDifference(['B8', 'B4']);

Map.setCenter(71, 51, 6);
Map.addLayer(ndvi, {min: -0.1, max: 0.6, palette: ['brown', 'yellow', 'green']}, 'NDVI');


// ------------------------------------------------------------
// STEP 2: NDVI change detection (2019 vs 2023)
// ------------------------------------------------------------
var region2 = ee.Geometry.Rectangle([67, 44, 76, 50]);

var ndvi2019 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region2)
  .filterDate('2019-06-01', '2019-08-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median()
  .normalizedDifference(['B8', 'B4']);

var ndvi2023 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region2)
  .filterDate('2023-06-01', '2023-08-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median()
  .normalizedDifference(['B8', 'B4']);

var diff = ndvi2023.subtract(ndvi2019);

Map.addLayer(diff, {
  min: -0.3,
  max: 0.3,
  palette: ['red', 'orange', 'white', 'lightgreen', 'green']
}, 'NDVI Change 2019-2023');

var stats = diff.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: region2,
  scale: 1000,
  maxPixels: 1e9
});
print('Mean NDVI change:', stats.get('nd'));


// ------------------------------------------------------------
// STEP 3: Multi-feature extraction (NDVI, BSI, NDWI, EVI)
// ------------------------------------------------------------
var region3 = ee.Geometry.Rectangle([67, 44, 76, 50]); // Betpak-Dala

var image3 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region3)
  .filterDate('2023-06-01', '2023-08-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

var ndvi3 = image3.normalizedDifference(['B8', 'B4']).rename('NDVI');

var bsi = image3.expression(
  '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
  {SWIR: image3.select('B11'), RED: image3.select('B4'),
   NIR: image3.select('B8'), BLUE: image3.select('B2')})
  .rename('BSI');

var ndwi = image3.normalizedDifference(['B3', 'B8']).rename('NDWI');

var evi = image3.expression(
  '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
  {NIR: image3.select('B8'), RED: image3.select('B4'),
   BLUE: image3.select('B2')})
  .rename('EVI');

var features = ndvi3.addBands(bsi).addBands(ndwi).addBands(evi);

Map.addLayer(ndvi3, {min: -0.1, max: 0.6, palette: ['brown', 'yellow', 'green']}, 'NDVI');
Map.addLayer(bsi, {min: -0.5, max: 0.5, palette: ['green', 'white', 'brown']}, 'BSI');

print('Feature bands:', features.bandNames());


// ------------------------------------------------------------
// STEP 4: ESA WorldCover labels (ground truth land classes)
// ------------------------------------------------------------
var worldcover = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .clip(region3);

Map.addLayer(worldcover, {}, 'WorldCover');
print('WorldCover loaded', worldcover.bandNames());

// WorldCover class codes:
// 10 = Tree cover
// 20 = Shrubland
// 30 = Grassland
// 40 = Cropland
// 50 = Built-up
// 60 = Bare/sparse vegetation  <-- "degraded land" proxy
// 80 = Water
// 90 = Herbaceous wetland
// 100 = Moss/lichen


// ------------------------------------------------------------
// STEP 5: Settlement distance feature (GHSL built-up layer)
// ------------------------------------------------------------
var settlement = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface')
  .clip(region3);

var settlementMask = settlement.gt(0.1);

var distanceToSettlement = settlementMask
  .fastDistanceTransform(256)
  .sqrt()
  .multiply(ee.Image.pixelArea().sqrt())
  .rename('dist_settlement');

Map.addLayer(settlement, {min: 0, max: 1, palette: ['white', 'red']}, 'Settlements');
Map.addLayer(distanceToSettlement, {min: 0, max: 50000, palette: ['red', 'yellow', 'white']}, 'Distance to settlement');


// ------------------------------------------------------------
// STEP 6: FULL PIPELINE - Combine everything + export samples
// (This version: Betpak-Dala region)
// ------------------------------------------------------------
var region_betpak = ee.Geometry.Rectangle([67, 44, 76, 50]);

var img_betpak = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region_betpak)
  .filterDate('2023-06-01', '2023-08-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

var ndvi_b = img_betpak.normalizedDifference(['B8', 'B4']).rename('NDVI');
var bsi_b = img_betpak.expression(
  '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
  {SWIR: img_betpak.select('B11'), RED: img_betpak.select('B4'),
   NIR: img_betpak.select('B8'), BLUE: img_betpak.select('B2')})
  .rename('BSI');
var ndwi_b = img_betpak.normalizedDifference(['B3', 'B8']).rename('NDWI');
var evi_b = img_betpak.expression(
  '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
  {NIR: img_betpak.select('B8'), RED: img_betpak.select('B4'),
   BLUE: img_betpak.select('B2')})
  .rename('EVI');

var settlement_b = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface')
  .clip(region_betpak);
var settlementMask_b = settlement_b.gt(0.1);
var distanceToSettlement_b = settlementMask_b
  .fastDistanceTransform(256)
  .sqrt()
  .multiply(ee.Image.pixelArea().sqrt())
  .rename('dist_settlement');

var worldcover_b = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .clip(region_betpak)
  .rename('label');

var combined_b = ndvi_b.addBands(bsi_b).addBands(ndwi_b).addBands(evi_b)
  .addBands(distanceToSettlement_b).addBands(worldcover_b);

var samples_b = combined_b.stratifiedSample({
  numPoints: 500,
  classBand: 'label',
  region: region_betpak,
  scale: 100,
  seed: 42,
  geometries: false
});

print('Betpak-Dala sample count:', samples_b.size());

Export.table.toDrive({
  collection: samples_b,
  description: 'betpak_samples_v2',
  fileFormat: 'CSV'
});


// ------------------------------------------------------------
// STEP 7: SAME PIPELINE for Zhanaarka district
// (different bounding box, everything else identical)
// ------------------------------------------------------------
var region_zh = ee.Geometry.Rectangle([69, 47, 74, 50]); // Zhanaarka district

var img_zh = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region_zh)
  .filterDate('2023-06-01', '2023-08-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

var ndvi_zh = img_zh.normalizedDifference(['B8', 'B4']).rename('NDVI');
var bsi_zh = img_zh.expression(
  '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
  {SWIR: img_zh.select('B11'), RED: img_zh.select('B4'),
   NIR: img_zh.select('B8'), BLUE: img_zh.select('B2')})
  .rename('BSI');
var ndwi_zh = img_zh.normalizedDifference(['B3', 'B8']).rename('NDWI');
var evi_zh = img_zh.expression(
  '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
  {NIR: img_zh.select('B8'), RED: img_zh.select('B4'),
   BLUE: img_zh.select('B2')})
  .rename('EVI');

var settlement_zh = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface')
  .clip(region_zh);
var settlementMask_zh = settlement_zh.gt(0.1);
var distanceToSettlement_zh = settlementMask_zh
  .fastDistanceTransform(256)
  .sqrt()
  .multiply(ee.Image.pixelArea().sqrt())
  .rename('dist_settlement');

var worldcover_zh = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .clip(region_zh)
  .rename('label');

var combined_zh = ndvi_zh.addBands(bsi_zh).addBands(ndwi_zh).addBands(evi_zh)
  .addBands(distanceToSettlement_zh).addBands(worldcover_zh);

var samples_zh = combined_zh.stratifiedSample({
  numPoints: 500,
  classBand: 'label',
  region: region_zh,
  scale: 100,
  seed: 42,
  geometries: false
});

print('Zhanaarka sample count:', samples_zh.size());

Export.table.toDrive({
  collection: samples_zh,
  description: 'zhanaarka_samples',
  fileFormat: 'CSV'
});

Map.setCenter(71.5, 48.5, 7);

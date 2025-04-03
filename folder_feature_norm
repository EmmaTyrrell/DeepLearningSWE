## normalizing features

feature_folder = "D:/ASOML/Sierras/features/"
scaled_folder = "D:/ASOML/Sierras/features/scaled/"

for filename in os.listdir(feature_folder):
    if filename.lower().endswith(".tif"):
        full_path = os.path.join(feature_folder, filename)
        print(full_path)
        # --- Read and scale ---
        with rasterio.open(full_path) as src:
            data = src.read(1).astype(np.float32)  # read first band
            profile = src.profile
        
            # Optional: Handle NoData
            nodata = profile.get("nodata")
            if nodata is not None:
                data[data == nodata] = np.nan
        
            scaled = min_max_scale(data)
        
            # Optional: set new NoData value (e.g. -9999)
            new_nodata = -9999
            scaled[np.isnan(scaled)] = new_nodata
        
            # --- Update metadata ---
            profile.update(
                dtype="float32",
                nodata=new_nodata,
                compress="lzw"
            )
        
            # --- Save output ---
            with rasterio.open(scaled_folder + f"{filename[:-4]}_scl.tif", "w", **profile) as dst:
                dst.write(scaled, 1)

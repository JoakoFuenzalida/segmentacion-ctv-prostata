%% INVENTARIO DE PACIENTES - DATASET VIÑA DEL MAR  (v2)
% Recorre cada archivo y lo clasifica por su campo Modality, en vez de usar
% dicomCollection. Es más lento, pero no se pierde archivos cuyos metadatos
% estén incompletos y no se cae si un paciente viene distinto al resto.
% No modifica ni guarda nada: solo imprime en la consola.

clear; clc;

carpeta_raiz = uigetdir(pwd, 'Selecciona la carpeta que contiene los pacientes');
if isequal(carpeta_raiz, 0); error('No se seleccionó carpeta.'); end

pacientes = dir(carpeta_raiz);
pacientes = pacientes([pacientes.isdir] & ~startsWith({pacientes.name}, '.'));
fprintf('Carpeta raíz: %s\nPacientes encontrados: %d\n', carpeta_raiz, numel(pacientes));

todos_los_roi = {};

for i = 1:numel(pacientes)
    id   = pacientes(i).name;
    ruta = fullfile(carpeta_raiz, id);
    fprintf('\n==================== Paciente %s ====================\n', id);

    archivos = dir(fullfile(ruta, '**', '*'));
    archivos = archivos(~[archivos.isdir]);
    peso_mb  = sum([archivos.bytes]) / 1e6;

    % --- Clasificar cada archivo segun su Modality ---
    ct_files = {}; rt_files = {}; otras = {}; no_dicom = 0;
    for k = 1:numel(archivos)
        f = fullfile(archivos(k).folder, archivos(k).name);
        try
            info = dicominfo(f, 'UseVRHeuristic', false);
        catch
            no_dicom = no_dicom + 1;
            continue;
        end
        if ~isfield(info, 'Modality'); continue; end
        switch upper(info.Modality)
            case 'CT';       ct_files{end+1} = f;   %#ok<SAGROW>
            case 'RTSTRUCT'; rt_files{end+1} = f;   %#ok<SAGROW>
            otherwise;       otras{end+1} = info.Modality; %#ok<SAGROW>
        end
    end

    fprintf('Archivos: %d (%.1f MB) | CT: %d | RTSTRUCT: %d | no-DICOM: %d\n', ...
        numel(archivos), peso_mb, numel(ct_files), numel(rt_files), no_dicom);
    if ~isempty(otras)
        fprintf('Otras modalidades presentes: %s\n', strjoin(unique(otras), ', '));
    end

    % --- Metadatos de la serie CT ---
    if ~isempty(ct_files)
        ict = dicominfo(ct_files{1});
        fprintf('Cortes CT: %d | Matriz: %d x %d\n', numel(ct_files), ict.Rows, ict.Columns);
        if isfield(ict, 'PixelSpacing')
            fprintf('PixelSpacing: [%.3f %.3f] mm', ict.PixelSpacing(1), ict.PixelSpacing(2));
        end
        if isfield(ict, 'SliceThickness')
            fprintf(' | SliceThickness: %.2f mm', ict.SliceThickness);
        end
        fprintf('\n');
        if isfield(ict, 'PatientPosition')
            fprintf('Posición del paciente: %s\n', ict.PatientPosition);
        end
        if isfield(ict, 'RescaleSlope')
            fprintf('RescaleSlope: %g | RescaleIntercept: %g\n', ...
                ict.RescaleSlope, ict.RescaleIntercept);
        else
            fprintf('ATENCION: sin RescaleSlope/Intercept en la cabecera.\n');
        end
        peso_hu = double(ict.Rows) * double(ict.Columns) * numel(ct_files) * 2 / 1e6;
        fprintf('Matriz HU estimada (int16): %.1f MB\n', peso_hu);
    else
        fprintf('ATENCION: este paciente no tiene archivos CT.\n');
    end

    % --- ROI del RTSTRUCT ---
    if isempty(rt_files)
        fprintf('ATENCION: no se encontró RTSTRUCT. Sin máscaras, este paciente no sirve para entrenar.\n');
    else
        if numel(rt_files) > 1
            fprintf('Hay %d RTSTRUCT; se listan los ROI de cada uno.\n', numel(rt_files));
        end
        for r = 1:numel(rt_files)
            try
                irt = dicominfo(rt_files{r});
                cont = dicomContours(irt);
                nombres = cont.ROIs.Name;
                fprintf('RTSTRUCT %d -> %d ROI:\n', r, numel(nombres));
                for k = 1:numel(nombres)
                    fprintf('   - %s\n', nombres{k});
                end
                todos_los_roi = [todos_los_roi; nombres(:)]; %#ok<AGROW>
            catch ME
                fprintf('No se pudo leer el RTSTRUCT %d: %s\n', r, ME.message);
            end
        end
    end
end

% --- Resumen: en cuántos pacientes aparece cada nombre de ROI ---
fprintf('\n==================== RESUMEN DE NOMBRES DE ROI ====================\n');
if isempty(todos_los_roi)
    fprintf('No se leyó ningún ROI.\n');
else
    [nombres_unicos, ~, idx] = unique(todos_los_roi);
    conteo = accumarray(idx, 1);
    for k = 1:numel(nombres_unicos)
        fprintf('%-35s %d vez/veces\n', nombres_unicos{k}, conteo(k));
    end
end

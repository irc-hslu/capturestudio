from capture.session import SyncedSession
from utils.misc import log

if __name__ == '__main__':
    # Format: <session_name>|<calib_session_name>|<recon_t_start>|<recon_t_total>;...;...
    SYNC_SESSIONS_ = [
        # 'Cagliari_2_5cams_Calib_1|-|400|2500', #
        # 'Cagliari_2_5cams_Calib_2|-|400|2200', #
        'Cagliari_2_5cams_Perf_1|Cagliari_2_5cams_Calib_2|400|8500',
        # 'Cagliari_1_Perf_5|Cagliari_1_Calib_6|0|0', # 1st brother: Leonardo started the timer 10 seconds after recording start. + 1min 40 sec offset
        # 'Cagliari_1_Perf_5|Cagliari_1_Calib_6|3390|300', # 1st brother: Test for 300 frames
        # 'Cagliari_1_Perf_5|Cagliari_1_Calib_6|360|8400', # 1st brother: All frames (after 10sec warmup)
        # 'Cagliari_1_Perf_7|Cagliari_1_Calib_6|300|5000', # 2nd brother: Leonardo started the timer 10 seconds after recording start. + 1min 40 sec offset
        # 'Cagliari_1_Perf_7|Cagliari_1_Calib_6|3330|300', # 2nd brother: Test for 300 frames
        # 'Cagliari_1_Perf_7|Cagliari_1_Calib_6|0|8000',  # 2nd brother: All frames (after 10sec warmup)
    ]
    RECON_CAMS_ = [1, 2, 3, 4, 5]

    submitted_job_ids = []
    for session_name_, calibration_session_name_, recon_start_frame_, recon_total_frames_ in [_.split('|') for _ in SYNC_SESSIONS_]:
        log(f'Processing Session: {session_name_} (calibration session: {calibration_session_name_})', 'info')

        job = (
            SyncedSession(session_name_, excel_sheet='Cagliari_Jun_2026')
                .download_from_nas()
                .synchronize(
                    trim_start_frame=int(recon_start_frame_) if int(recon_start_frame_) > 0 else None,
                    trim_total_frames=int(recon_total_frames_) if int(recon_total_frames_) > 0 else None
                )
        )

        # job = SyncedSession(session_name_, excel_sheet='Cagliari_Nov_2025')

        if calibration_session_name_ != '-':
            job = job.preprocess(interactive_annotation=True)
            pass
            # job = (job
            #        .preprocess()
            #        .reconstruct(
            #             calibration_session_name=calibration_session_name_,
            #             start_frame=int(recon_start_frame_),
            #             total_frames=int(recon_total_frames_),
            #             cam_idx=RECON_CAMS_,
            #             force=True,
            #             rotate=None,
            #             orbit_type='audience',
            #             camera_orbit_velocity=1.0,
            #             save_ply=False
            #        )
            # )
        else:
            job = job.calibrate(calibration_method=job.excel_data['Subject'])

        job.upload(delete_capturestudio_cache=False)

        submitted_job_ids.append(
            job
                .to_celery(export_graph=False, export_format='pdf')
                .submit()
        )

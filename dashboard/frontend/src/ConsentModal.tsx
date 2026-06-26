import { useState } from "react";

type Props = {
  onAccept: () => void;
  onDecline: () => void;
};

export default function ConsentModal({ onAccept, onDecline }: Props) {
  const [understood, setUnderstood] = useState(false);
  const [agree, setAgree] = useState(false);

  const canSubmit = understood && agree;

  return (
    <div className="consent-overlay">
      <div className="consent-modal">
        <div className="consent-header">
          <span className="consent-badge">Privacy Notice</span>
          <h2>Data Collection Consent</h2>
          <p className="consent-lede">
            Before you begin recording, please review how your data will be handled.
          </p>
        </div>

        <div className="consent-body">
          <section>
            <h3>What we collect</h3>
            <ul>
              <li>
                <strong>RGB video</strong> from your webcam while you perform signs
              </li>
              <li>
                <strong>3D skeleton keypoints</strong> extracted from the video (hand
                poses, body landmarks, facial landmarks)
              </li>
            </ul>
          </section>

          <section className="consent-highlight">
            <h3>Privacy guarantee</h3>
            <ul>
              <li>
                <span className="consent-check">✓</span>{" "}
                <strong>RGB video data will NOT be made public.</strong> It stays on
                this server and is used only to extract skeleton keypoints.
              </li>
              <li>
                <span className="consent-check">✓</span>{" "}
                <strong>Skeleton keypoints</strong> (landmark positions only — no
                identifiable imagery) may be published as part of an open dataset
                or included in research outputs.
              </li>
              <li>
                <span className="consent-check">✓</span> You can{" "}
                <strong>withdraw consent</strong> at any time, which will stop
                collection and delete your recorded data.
              </li>
            </ul>
          </section>

          <section>
            <h3>How your data is used</h3>
            <ul>
              <li>Training and evaluating sign language recognition models</li>
              <li>Academic research in computer vision and linguistics</li>
              <li>Improving our open-source sign language tools</li>
            </ul>
          </section>

          <label className="consent-checkbox">
            <input
              type="checkbox"
              checked={understood}
              onChange={(e) => setUnderstood(e.target.checked)}
            />
            <span>
              I understand what data is collected and how it will be used
            </span>
          </label>

          <label className="consent-checkbox">
            <input
              type="checkbox"
              checked={agree}
              onChange={(e) => setAgree(e.target.checked)}
            />
            <span>
              I consent to the collection of my video data and agree to the
              skeleton keypoints being made publicly available
            </span>
          </label>
        </div>

        <div className="consent-footer">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onDecline}
          >
            Decline
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!canSubmit}
            onClick={onAccept}
          >
            Accept &amp; Start Collection
          </button>
        </div>
      </div>
    </div>
  );
}

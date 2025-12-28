# n8n GitHub Issue: Webhook Path Corruption After Workflow Duplication

## Title
**Duplicated workflow hijacks original workflow's webhook path in webhook_entity table despite having different webhookId**

---

## Describe the bug

When duplicating a workflow containing a webhook trigger, the duplicated workflow correctly receives a **new unique webhookId** in its node definition. However, during activation, n8n incorrectly registers the **original workflow's webhook path** pointing to the **duplicated workflow** in the `webhook_entity` table.

This causes all incoming requests to the original webhook URL to be routed to the duplicated (often inactive) workflow instead of the original active workflow.

The corruption:
1. Persists across n8n restarts
2. Is actively re-created during n8n startup activation sequence
3. Cannot be fixed by database corrections alone (n8n overwrites fixes on restart)
4. Is not caught by the conflict detection added in PR #14783 / v1.91.0

---

## Environment

| Component | Version/Details |
|-----------|-----------------|
| n8n version | 1.123.5 |
| Database | PostgreSQL 15 |
| Running | SystemD service (self-hosted) |
| Node.js | v20.x |
| OS | Debian 12 |

---

## Steps to reproduce

### Initial Setup
1. Create a workflow with a Webhook trigger node
2. Activate the workflow
3. Verify webhook is registered correctly:
```sql
SELECT "webhookPath", "workflowId" FROM webhook_entity;
-- Shows: original-uuid -> original-workflow-id
```

### Trigger the Bug
4. Duplicate the workflow using n8n's "Duplicate" feature
5. The duplicated workflow receives a NEW webhookId (visible in node parameters)
6. Save the duplicated workflow (do NOT activate it)
7. Restart n8n service

### Observe the Corruption
8. Check webhook_entity table:
```sql
SELECT "webhookPath", "workflowId", w.name, w.active
FROM webhook_entity we
JOIN workflow_entity w ON we."workflowId" = w.id;
```

**Expected result:**
```
original-webhook-uuid | original-workflow-id | Original Workflow | t
```

**Actual result:**
```
original-webhook-uuid | duplicated-workflow-id | Duplicated Workflow | f
```

The original workflow's webhook path now points to the duplicated workflow!

---

## Detailed Technical Analysis

### Database State Before Bug

**workflow_entity:**
| id | name | active | webhook node path |
|----|------|--------|-------------------|
| yDbfhooKemfhMIkC | SAI Webhook + YOLO | true | e861ad7c-8160-4964-8953-5e3a02657293 |

**webhook_entity:**
| webhookPath | workflowId |
|-------------|------------|
| e861ad7c-8160-4964-8953-5e3a02657293 | yDbfhooKemfhMIkC |

### After Duplication (Dec 16, 2025)

**workflow_entity:**
| id | name | active | webhook node path |
|----|------|--------|-------------------|
| yDbfhooKemfhMIkC | SAI Webhook + YOLO | true | e861ad7c-8160-4964-8953-5e3a02657293 |
| mzurJ2b8CsGhaFYo | Gemelo (duplicate) | false | 6a1f3735-9271-48b3-b5b0-c9cd50185c1f |

Note: The duplicated workflow has a **DIFFERENT** webhookId in its node definition!

**webhook_entity (CORRUPTED):**
| webhookPath | workflowId |
|-------------|------------|
| e861ad7c-8160-4964-8953-5e3a02657293 | mzurJ2b8CsGhaFYo | ← WRONG! |

### Key Observation

The duplicated workflow's node definition contains the correct NEW path:
```json
{
  "id": "c920b6ad-c12f-4451-8da6-0644a883a166",
  "name": "Webhook",
  "type": "n8n-nodes-base.webhook",
  "webhookId": "6a1f3735-9271-48b3-b5b0-c9cd50185c1f",
  "parameters": {
    "path": "6a1f3735-9271-48b3-b5b0-c9cd50185c1f"
  }
}
```

But n8n registers the ORIGINAL path (`e861ad7c...`) for this workflow in webhook_entity!

---

## n8n Startup Logs

During every restart, n8n "activates" the duplicated workflow even though `active = false`:

```
Dec 26 22:24:30 n8n[3637059]: Start Active Workflows:
Dec 26 22:24:30 n8n[3637059]: Activated workflow "SAI Webhook + YOLO" (ID: yDbfhooKemfhMIkC)
...
Dec 26 22:24:31 n8n[3637059]: Activated workflow "Gemelo" (ID: mzurJ2b8CsGhaFYo)  ← active=false but still "Activated"!
```

This activation process corrupts the webhook_entity table on every restart.

---

## Impact

- **3 days of production outage** (Dec 23-26, 2025)
- **~4,500 webhook requests** routed to wrong (inactive) workflow
- Original workflow received **zero** requests despite being active
- ETL pipeline stopped logging executions
- Alert system stopped functioning

---

## Attempted Fixes That Failed

### 1. Database Correction (Failed)
```sql
DELETE FROM webhook_entity WHERE "webhookPath" = 'e861ad7c-...';
INSERT INTO webhook_entity (...) VALUES ('e861ad7c-...', 'yDbfhooKemfhMIkC');
```
**Result:** n8n overwrites correction on restart

### 2. Set triggerCount = 0 on Duplicate (Failed)
```sql
UPDATE workflow_entity SET "triggerCount" = 0 WHERE id = 'mzurJ2b8CsGhaFYo';
```
**Result:** n8n resets triggerCount to 1 on restart and still activates

### 3. Ensure active = false (Failed)
The workflow was already `active = false` but n8n still "Activated" it during startup.

### 4. Only Working Fix: Delete Duplicate Workflow
```sql
DELETE FROM workflow_entity WHERE id = 'mzurJ2b8CsGhaFYo';
-- Then fix webhook_entity
```
**Result:** Finally worked - no more corruption on restart

---

## Related Issues

| Issue | Relationship |
|-------|-------------|
| #2386 | Duplicate keeps same URL (by design) - different from this bug |
| #11966 | Copying webhook keeps old UUID - similar but that issue was closed as stale |
| #14809 | Duplicate paths allowed - fixed in v1.91.0 but doesn't cover this case |
| #14783 | Webhook takeover prevention - only handles same-path conflicts |

This bug is **NOT covered** by the fix in PR #14783 because:
- PR #14783 handles workflows with **identical paths** competing
- This bug involves a workflow with a **different path** in its definition somehow registering with the **wrong path**

---

## Expected behavior

1. Duplicated workflows should receive unique webhookIds (✓ this works)
2. The webhookId in node definition should match what's registered in webhook_entity (✗ broken)
3. Workflows with `active = false` should NOT be "Activated" during startup (✗ broken)
4. webhook_entity should only contain entries for workflows where `active = true` (✗ broken)

---

## Suggested Fix

1. **During duplication, do NOT copy the original's webhook registration** - each workflow should only register its own webhookId from its node definition
2. **During workflow activation, validate that the path being registered matches the path in the workflow's node definition** - reject mismatches
3. **Do not register webhooks for workflows where `active = false`** during startup sequence
4. **Add integrity check**: `webhook_entity.webhookPath` must match the corresponding workflow's webhook node `parameters.path`
5. **Use unique constraint** on `webhook_entity.webhookPath` to prevent silent overwrites

---

## Workaround

### Option 1: Delete the Duplicated Workflow (Most Reliable)

```sql
-- Backup first
CREATE TABLE workflow_backup AS SELECT * FROM workflow_entity WHERE id = 'duplicated-id';

-- Delete shared_workflow link first
DELETE FROM shared_workflow WHERE "workflowId" = 'duplicated-id';

-- Delete the problematic duplicate
DELETE FROM workflow_entity WHERE id = 'duplicated-id';

-- Fix webhook_entity to point back to original
UPDATE webhook_entity
SET "workflowId" = 'original-workflow-id'
WHERE "webhookPath" = 'original-path';

-- Restart n8n
sudo systemctl restart n8n
```

### Option 2: Avoid Duplicating Webhook Workflows

Until this bug is fixed, **do not use the Duplicate feature** on workflows containing webhook triggers. Instead:
1. Export the workflow as JSON
2. Import it as a new workflow
3. Manually update the webhook path in the imported workflow
4. This ensures the new workflow gets its own unique webhook registration

---

## Bug Reproduction - CONFIRMED (Dec 27, 2025)

### ✅ Reproducible Test Case (Clean Room)

**Test Date:** December 27, 2025
**n8n Version:** 1.123.5

#### Step 1: Create New Workflow with Webhook

1. Created new workflow `Test-Webhook-Bug`
2. Added Webhook node with custom path: `test-bug-reproduction`
3. Activated the workflow

**Database State After Step 1:**
```sql
-- workflow_entity
id               | name             | active | activeVersionId
-----------------+------------------+--------+--------------------------------------
AY8Cig5S6wzsWFOW | Test-Webhook-Bug | t      | 98426fad-4e32-4b88-8332-ac8c0ec36b24

-- workflow_history
versionId                            | workflowId       | history_path
-------------------------------------+------------------+-----------------------
98426fad-4e32-4b88-8332-ac8c0ec36b24 | AY8Cig5S6wzsWFOW | test-bug-reproduction  ✓

-- webhook_entity
webhookPath           | workflowId       | name             | active
----------------------+------------------+------------------+--------
test-bug-reproduction | AY8Cig5S6wzsWFOW | Test-Webhook-Bug | t  ✓
```

#### Step 2: Duplicate the Workflow

1. Used n8n UI menu → **Duplicate**
2. Duplicate created as `Test-Webhook-Bug copy`
3. Duplicate was created as **inactive** (expected behavior)

**Database State After Step 2 - BUG VISIBLE:**
```sql
-- workflow_entity (NOTE: SAME activeVersionId!)
id               | name                  | active | activeVersionId
-----------------+-----------------------+--------+--------------------------------------
AY8Cig5S6wzsWFOW | Test-Webhook-Bug      | t      | 98426fad-4e32-4b88-8332-ac8c0ec36b24
01zK96fRNfHodYYC | Test-Webhook-Bug copy | f      | 98426fad-4e32-4b88-8332-ac8c0ec36b24  ← BUG!

-- workflow_history (duplicate has its OWN entry with NEW path)
versionId                            | workflowId       | history_path
-------------------------------------+------------------+--------------------------------------
98426fad-4e32-4b88-8332-ac8c0ec36b24 | AY8Cig5S6wzsWFOW | test-bug-reproduction  (original)
34221a46-4cbd-47f4-80f4-d56ff0d4d0a0 | 01zK96fRNfHodYYC | 26f1da72-102c-4088-a836-ee104990ce83  (duplicate - NEW path)
```

**🔴 THE BUG:** Duplicate's `activeVersionId` points to ORIGINAL's `workflow_history` entry, not its own!

#### Step 3: Restart n8n

```bash
sudo systemctl restart n8n
```

**Database State After Restart - WEBHOOK HIJACKED:**
```sql
-- webhook_entity
webhookPath           | workflowId       | name                  | active
----------------------+------------------+-----------------------+--------
test-bug-reproduction | 01zK96fRNfHodYYC | Test-Webhook-Bug copy | f  ← HIJACKED!
```

**The INACTIVE duplicate has stolen the webhook from the ACTIVE original!**

---

## Root Cause - CONFIRMED

### The Exact Bug Location

**File:** Workflow duplication logic (likely `packages/cli/src/workflows/workflows.controller.ts` or frontend)

**Bug:** When duplicating a workflow, the `activeVersionId` field is **copied from the original** instead of being set to `null` or to the duplicate's own `workflow_history.versionId`.

### Why This Causes Webhook Hijacking

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DUPLICATION BUG FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: Original workflow active                                        │
│  ┌──────────────────────┐     ┌──────────────────────┐                  │
│  │ workflow_entity      │     │ workflow_history     │                  │
│  │ id: ORIGINAL         │────▶│ versionId: AAA       │                  │
│  │ activeVersionId: AAA │     │ path: "my-webhook"   │                  │
│  └──────────────────────┘     └──────────────────────┘                  │
│                                                                          │
│  STEP 2: Duplicate created (BUG OCCURS HERE)                            │
│  ┌──────────────────────┐     ┌──────────────────────┐                  │
│  │ workflow_entity      │     │ workflow_history     │                  │
│  │ id: DUPLICATE        │──┐  │ versionId: BBB       │  ← IGNORED!      │
│  │ activeVersionId: AAA │  │  │ path: "new-uuid"     │                  │
│  └──────────────────────┘  │  └──────────────────────┘                  │
│                            │                                             │
│                            │  ┌──────────────────────┐                  │
│                            └─▶│ workflow_history     │  ← WRONG LINK!   │
│                               │ versionId: AAA       │                  │
│                               │ path: "my-webhook"   │  (ORIGINAL's)    │
│                               └──────────────────────┘                  │
│                                                                          │
│  STEP 3: On restart, n8n activates DUPLICATE with ORIGINAL's path       │
│  - Reads activeVersionId: AAA                                            │
│  - Loads nodes from workflow_history AAA (ORIGINAL's nodes!)            │
│  - Registers webhook "my-webhook" for DUPLICATE                          │
│  - Upsert overwrites → ORIGINAL loses its webhook                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### This Explains ALL Previously Observed Symptoms

| Symptom | Explanation |
|---------|-------------|
| Duplicate shows NEW webhookId in editor | `workflow_entity.nodes` has the new ID (frontend works correctly) |
| Duplicate registers with ORIGINAL's path | `activeVersionId` points to original's `workflow_history` |
| Bug only occurs after restart | Webhooks are registered from `activeVersion` during startup |
| Inactive workflow hijacks active's webhook | Activation order + silent upsert = last one wins |
| Database fixes don't persist | n8n re-registers from wrong `activeVersionId` on every restart |

### Definitive Fix Required

**In the duplication logic, change:**
```typescript
// CURRENT (BUGGY):
duplicate.activeVersionId = original.activeVersionId;  // Copies wrong reference!

// FIXED:
duplicate.activeVersionId = null;  // Or: newWorkflowHistory.versionId
```

---

## Fix Status - VERIFIED IN SOURCE CODE

### Bug Location Identified

**File:** `packages/frontend/editor-ui/src/app/components/DuplicateWorkflowDialog.vue`

### Version Comparison

**n8n@1.123.5 (BUGGY - our production):**
```typescript
// Lines 92-100
const {
    createdAt,
    updatedAt,
    usedCredentials,
    id,
    homeProject,
    sharedWithProjects,
    ...workflow  // ← activeVersionId COPIED TO DUPLICATE!
} = await workflowsStore.fetchWorkflow(props.data.id);
```

**master (FIXED - PR #23113 + PR #23166):**
```typescript
// Lines 93-105
const {
    createdAt,
    updatedAt,
    usedCredentials,
    id,
    homeProject,
    sharedWithProjects,
    activeVersionId,  // ← NOW REMOVED ✓
    activeVersion,    // ← NOW REMOVED ✓
    active,           // ← NOW REMOVED ✓
    ...workflow
} = await workflowsStore.fetchWorkflow(props.data.id);
```

### Fix Details

| Aspect | Details |
|--------|---------|
| **Primary Fix PR** | [#23113](https://github.com/n8n-io/n8n/pull/23113) |
| **Commit** | `883c409b` |
| **Date** | December 12, 2025 |
| **Title** | "fix: Make sure duplicating workflows creates them as unpublished" |
| **Author** | @dariacodes (Daria) |
| **Secondary PR** | [#23166](https://github.com/n8n-io/n8n/pull/23166) (Dec 24, 2025) - Additional refinements |
| **Fix** | Added `activeVersionId`, `activeVersion`, `active` to destructuring exclusion list |

### Upgrade Recommendation

**Upgrade to n8n >= 2.0.1** (first version after Dec 12 fix) to get the complete fix.

Until then, the workaround is to **avoid using the Duplicate feature** on workflows with webhook triggers.

---

## Bug Lifecycle Investigation (Dec 27, 2025)

### Complete Timeline

| Date | Event | Details |
|------|-------|---------|
| **Nov 20, 2025** | Bug Introduced | PR #21202 added `activeVersionId` to workflow versioning |
| **Dec 1, 2025** | Feature expanded | PR #22533 deprecated `workflow.active` field |
| **Dec 5, 2025** | Feature expanded | PR #22805 added `activeVersionId` to broadcast events |
| **Dec 8, 2025** | v2.0.0 Released | **Bug included in major release** |
| **Dec 12, 2025** | Bug Fixed | PR #23113 explicitly fixes duplication issue |
| **Dec 24, 2025** | Fix refined | PR #23166 adds additional fixes for sub-workflows |

**Bug existed for: ~22 days (Nov 20 - Dec 12, 2025)**

### Was the Fix Intentional?

**YES** - The fix was explicitly intentional:

1. **Same Author**: Daria (@dariacodes) authored both:
   - PR #21202 (Nov 20) - Introduced `activeVersionId` field
   - PR #23113 (Dec 12) - Fixed the duplication bug

2. **Explicit Issue Reference**: PR #23113 closes internal issue "ADO-4533"

3. **PR Description** explicitly states:
   > "This PR addresses two issues when duplicating published workflows:
   > 1. Frontend problem: Sending a create payload with all active fields set
   > 2. Backend problem: Ignoring `active` and `activeVersionId` fields, but not `activeVersion`"

### Related Community Issues

Multiple community reports existed but were **not properly addressed**:

| Issue | Date | Status | Relationship |
|-------|------|--------|--------------|
| [#2386](https://github.com/n8n-io/n8n/issues/2386) | Oct 2021 | Closed as "expected behavior" | Same webhook URL duplication symptom |
| [#10836](https://github.com/n8n-io/n8n/issues/10836) | Sep 2024 | Closed as "Not Planned" | Outdated duplicate workflow execution |
| [#11966](https://github.com/n8n-io/n8n/issues/11966) | Nov 2024 | Closed as "Not Planned" | Webhook UUID retention when copying |

### Bug Introduction Analysis

**PR #21202** (Nov 20, 2025) introduced the workflow versioning system with `activeVersionId`:

```
                     ┌─────────────────────────────────────────────────────┐
                     │              BUG INTRODUCTION TIMELINE               │
                     ├─────────────────────────────────────────────────────┤
                     │                                                      │
                     │  Nov 20     Dec 1      Dec 8      Dec 12    Dec 24  │
                     │    │          │          │           │         │    │
                     │    ▼          ▼          ▼           ▼         ▼    │
                     │  ┌───┐     ┌───┐      ┌─────┐    ┌─────┐   ┌─────┐  │
                     │  │PR │     │PR │      │v2.0 │    │ PR  │   │ PR  │  │
                     │  │#21│     │#22│      │.0   │    │#231 │   │#231 │  │
                     │  │202│     │533│      │     │    │13   │   │66   │  │
                     │  └───┘     └───┘      └─────┘    └─────┘   └─────┘  │
                     │    │                     │          │               │
                     │    └─── BUG WINDOW ──────┘          │               │
                     │    │      (22 days)      │          │               │
                     │    │                     │          │               │
                     │  BUG                v2.0.0        FIX              │
                     │  INTRODUCED         SHIPPED       MERGED           │
                     │                     WITH BUG                       │
                     │                                                      │
                     └─────────────────────────────────────────────────────┘
```

### Why v1.123.5 Has the Bug

The workflow versioning feature with `activeVersionId` was backported/included in late v1.x releases:

- **v1.123.5** (our production): Has `activeVersionId` field, bug present
- **v2.0.0** (Dec 8, 2025): Bug still present (fix came 4 days later)
- **v2.0.1+** (after Dec 12): Bug fixed

### n8n v2.0 Context

The `activeVersionId` field is central to n8n 2.0's new "Publish" workflow:
- **Save**: Preserves edits without changing production
- **Publish**: Explicitly pushes changes live
- **Draft versions**: Allow testing before publishing

This explains why the bug went unnoticed initially - the versioning feature was complex and the duplication edge case was missed.

### Conclusion

This was a **temporal regression bug** in the n8n v2.0 development cycle:
1. Introduced Nov 20, 2025 during workflow versioning implementation
2. Shipped in v2.0.0 on Dec 8, 2025
3. Fixed 4 days later on Dec 12, 2025
4. **Our production (v1.123.5) predates the fix but has the buggy versioning code**

### Failed Reproduction Methods (for reference)

Direct database insertion did not trigger the bug because:
- Workflows require `workflow_history` entries to be recognized by n8n
- The bug only occurs through the UI "Duplicate" feature which creates proper history entries

---

## Source Code Analysis

### Architecture Overview

n8n uses a dual-storage system for workflows:
1. **`workflow_entity.nodes`** - Current editable nodes (what you see in the editor)
2. **`workflow_history.nodes`** - Published/activated nodes (what runs for webhooks)

When a workflow is activated, n8n reads from `workflow_history.nodes`, NOT from `workflow_entity.nodes`.

### Key Source Files Analyzed

| File | Purpose |
|------|---------|
| `packages/cli/src/webhooks/webhook.service.ts` | Webhook registration logic |
| `packages/cli/src/active-workflow-manager.ts` | Workflow activation on startup |
| `packages/workflow/src/expression.ts` | Expression resolution (`getSimpleParameterValue`) |
| `packages/workflow/src/node-helpers.ts` | Path construction (`getNodeWebhookPath`) |
| `packages/workflow/src/workflow-data-proxy.ts` | Parameter resolution (`$parameter`) |
| `packages/frontend/editor-ui/src/app/composables/useWorkflowSaving.ts` | Frontend duplication logic |
| `packages/frontend/editor-ui/src/app/components/DuplicateWorkflowDialog.vue` | Duplicate UI dialog |

### Bug Flow Analysis

#### 1. Frontend Duplication (WORKS CORRECTLY)

**File:** `packages/frontend/editor-ui/src/app/composables/useWorkflowSaving.ts:300-313`
```typescript
if (resetWebhookUrls) {
    workflowDataRequest.nodes = workflowDataRequest.nodes!.map((node) => {
        if (node.webhookId) {
            const newId = nodeHelpers.assignWebhookId(node);  // ✓ New UUID generated
            if (!isExpression(node.parameters.path)) {
                node.parameters.path = newId;  // ✓ Path updated to new UUID
            }
            changedNodes[node.name] = node.webhookId;
        }
        return node;
    });
}
```

**Finding:** The frontend correctly generates a NEW webhookId and updates the path.

#### 2. Webhook Description Definition

**File:** `packages/nodes-base/nodes/Webhook/description.ts`
```typescript
export const defaultWebhookDescription: IWebhookDescription = {
    name: 'default',
    isFullPath: true,  // Returns path directly without prefix
    path: '={{$parameter["path"]}}',  // Expression that reads from node params
};
```

**Finding:** The webhook path uses expression `$parameter["path"]` which resolves to `node.parameters.path`.

#### 3. Path Resolution During Activation

**File:** `packages/cli/src/webhooks/webhook.service.ts:212-256`
```typescript
getNodeWebhooks(workflow: Workflow, node: INode, additionalData, ignoreRestartWebhooks) {
    // ...
    let nodeWebhookPath = workflow.expression.getSimpleParameterValue(
        node,
        webhookDescription.path,  // '={{$parameter["path"]}}'
        mode,
        {},
    );
    // ...
}
```

**Finding:** The path is resolved by evaluating `$parameter["path"]` against the workflow's nodes.

#### 4. Expression Resolution

**File:** `packages/workflow/src/workflow-data-proxy.ts:257-330`
```typescript
nodeParameterGetter(nodeName: string) {
    return new Proxy({}, {
        get: (target, name) => {
            const node = this.workflow.nodes[nodeName];
            // ...
            returnValue = node.parameters[name];
            return returnValue;
        },
    });
}
```

**Finding:** `$parameter["path"]` reads from `workflow.nodes[nodeName].parameters.path`.

#### 5. Workflow Construction During Startup (LIKELY BUG LOCATION)

**File:** `packages/cli/src/active-workflow-manager.ts:597-616`
```typescript
const { nodes, connections } = dbWorkflow.activeVersion;  // ← Uses activeVersion!
workflow = new Workflow({
    id: dbWorkflow.id,
    name: dbWorkflow.name,
    nodes,  // ← These are from workflow_history, not workflow_entity
    connections,
    // ...
});
```

**Finding:** During activation, n8n constructs the Workflow from `activeVersion.nodes` which are stored in `workflow_history`.

#### 6. Webhook Registration (SILENT OVERWRITE)

**File:** `packages/cli/src/webhooks/webhook.service.ts:128-138`
```typescript
async storeWebhook(webhook: WebhookEntity) {
    try {
        await this.cacheService.set(webhook.cacheKey, webhook);
    } catch (error) {
        // Error handling...
    }
    await this.webhookRepository.upsert(webhook, ['method', 'webhookPath']);
}
```

**Finding:** Uses `upsert` with conflict resolution on `['method', 'webhookPath']`. **If two workflows have the same path, the second one silently overwrites the first!**

### Root Cause Hypothesis

The bug occurs when:

1. **Original workflow** is active with `workflow_history.nodes[webhook].parameters.path = "abc-123"`
2. **Duplicate workflow** is created with frontend correctly generating NEW path `"xyz-789"`
3. **BUT:** During duplication/save, the `workflow_history` entry for the duplicate somehow contains the ORIGINAL path `"abc-123"` instead of the new path `"xyz-789"`
4. On restart, both workflows try to register webhook with path `"abc-123"`
5. The upsert silently overwrites, and whichever workflow activates LAST wins

### Specific Bug Location Candidates

1. **`workflow_history` creation during duplicate save** - The history entry may be copying the wrong node data
2. **`activeVersion` assignment** - The duplicate may be incorrectly linking to the original's activeVersion
3. **Activation order on restart** - Race condition where the duplicate activates after the original

### Evidence Supporting This Theory

From our reproduction test:
- Original workflow: `Sai-daily-test` has webhookId `8ecbf1c4-...` in BOTH `workflow_entity.nodes` AND `workflow_history.nodes`
- Duplicate workflow: `Sai-daily-test copy` has webhookId `8baa8a52-...` in `workflow_entity.nodes` BUT may have `8ecbf1c4-...` in its `workflow_history.nodes`

This would explain why:
1. The duplicate shows the correct NEW webhookId in the editor (from `workflow_entity`)
2. But registers with the ORIGINAL path (from `workflow_history`)

### Conflict Check (EXISTS BUT DOESN'T HELP)

**File:** `packages/frontend/editor-ui/src/app/composables/useWorkflowSaving.ts:351-364`
```typescript
if (workflowData.activeVersionId !== null) {
    const conflict = await checkConflictingWebhooks(workflowData.id);
    if (conflict) {
        workflowData.active = false;
        workflowData.activeVersionId = null;
        toast.showMessage({
            title: 'Conflicting Webhook Path',
            message: `Workflow set to inactive...`,
            type: 'error',
        });
    }
}
```

**Finding:** This check ONLY runs when `activeVersionId !== null`. Since duplicates are created as inactive, this check never triggers.

---

## Additional Context

- The duplication was performed via n8n UI "Duplicate" feature
- Both workflows use the same credentials (httpHeaderAuth)
- The corruption occurs on first n8n restart after duplication
- Production environment with ~1,400 webhook calls/day
- **Confirmed:** pinData is NOT the cause (both workflows had empty pinData in reproduction test)
- The bug is in n8n's core duplication/activation logic - it incorrectly associates the original's webhook path with the duplicate

---

## Recommended Fix Implementation

Based on the source code analysis, here are targeted fixes:

### Fix 1: Add Unique Constraint (Database Level)

```sql
-- Prevent silent overwrites
ALTER TABLE webhook_entity ADD CONSTRAINT webhook_path_unique UNIQUE (method, "webhookPath");
```

This will cause the upsert to fail loudly instead of silently overwriting.

### Fix 2: Validate Path Before Registration

**File:** `packages/cli/src/webhooks/webhook.service.ts`

```typescript
async storeWebhook(webhook: WebhookEntity) {
    // Check if path matches what's in the node definition
    const workflow = await this.getWorkflowById(webhook.workflowId);
    const node = workflow.nodes.find(n => n.name === webhook.node);
    if (node.parameters.path !== webhook.webhookPath) {
        throw new Error(`Path mismatch: node has ${node.parameters.path} but registering ${webhook.webhookPath}`);
    }

    await this.webhookRepository.upsert(webhook, ['method', 'webhookPath']);
}
```

### Fix 3: Ensure workflow_history Gets Updated Nodes

During workflow save after duplication, ensure the `workflow_history` entry contains the MODIFIED nodes (with new webhookIds) rather than the original nodes.

### Fix 4: Add Conflict Check on Startup

**File:** `packages/cli/src/active-workflow-manager.ts`

Before registering webhooks during startup, check if any other ACTIVE workflow is already using that path
